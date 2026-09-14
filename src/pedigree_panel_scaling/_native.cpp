#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr const char* capsule_name = "pedigree_native.exact_memo";
PyObject* limit_exception = nullptr;

struct PythonFailure {};
struct LimitFailure { std::string reason; };
struct Ref {
    PyObject* value;
    explicit Ref(PyObject* object) : value(object) { if (!value) throw PythonFailure{}; }
    ~Ref() { Py_DECREF(value); }
    Ref(const Ref&) = delete;
};

[[noreturn]] void invalid(const char* message) {
    PyErr_SetString(PyExc_ValueError, message);
    throw PythonFailure{};
}
PyObject* required(PyObject* mapping, const char* name) {
    PyObject* value = PyDict_GetItemString(mapping, name);
    if (!value) invalid("Native payload is missing a required field");
    return value;
}
double real(PyObject* value) {
    double result = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) throw PythonFailure{};
    if (!std::isfinite(result)) invalid("Native numerical inputs must be finite");
    return result;
}
uint64_t integer(PyObject* value) {
    uint64_t result = PyLong_AsUnsignedLongLong(value);
    if (PyErr_Occurred()) throw PythonFailure{};
    return result;
}
std::array<double, 3> triple(PyObject* value) {
    Ref items(PySequence_Fast(value, "Expected three numerical values"));
    if (PySequence_Fast_GET_SIZE(items.value) != 3) invalid("Expected three numerical values");
    std::array<double, 3> result;
    for (int i = 0; i < 3; ++i) result[i] = real(PySequence_Fast_GET_ITEM(items.value, i));
    return result;
}
unsigned bits(uint64_t maximum) {
    unsigned result = 0;
    while (maximum) { ++result; maximum >>= 1; }
    return result;
}
uint64_t low_mask(unsigned width) {
    return width == 64 ? UINT64_MAX : (width ? (uint64_t(1) << width)-1 : 0);
}

struct Sum {
    double value = 0.0, correction = 0.0;
    void add(double x) {
        double next = value+x;
        correction += std::abs(value) >= std::abs(x) ? (value-next)+x : (x-next)+value;
        value = next;
    }
    double total() const { return value+correction; }
};
struct Transition { double probability; uint32_t next; };
struct Locus {
    std::array<double, 3> affine, risk, negative;
    std::array<bool, 3> constant;
    std::array<std::vector<Transition>, 3> successors;
};
struct Result { double value; int role; };
using Memo = std::unordered_map<uint64_t, Result>;

struct State {
    std::vector<std::vector<Locus>> profiles;
    std::vector<unsigned> gene_profile, shifts, widths;
    std::vector<std::pair<size_t, size_t>> spans;
    std::vector<uint32_t> scratch;
    Memo memo;
    unsigned remaining_width = 0;
    uint32_t children = 0;
    double fixed = 0.0, variable = 0.0;
    uint64_t max_states = 0, max_transitions = 0;
    double max_seconds = 0.0;
    PyObject* progress = nullptr;
    uint64_t states = 0, transitions = 0, terminal = 0, independent_count = 0, ticks = 0;
    uint64_t root_key = 0;
    std::array<double, 4> root_scores{};
    std::array<bool, 4> root_present{};
    Clock::time_point started = Clock::now(), last_progress = started-std::chrono::seconds(1);

    ~State() { Py_XDECREF(progress); }
    double elapsed() const { return std::chrono::duration<double>(Clock::now()-started).count(); }

    void parse(PyObject* payload, PyObject* root, PyObject* limits, PyObject* callback) {
        if (!PyDict_Check(payload)) invalid("Native payload must be a dictionary");
        uint64_t count = integer(required(payload, "children"));
        if (!count || count > UINT32_MAX) invalid("Native child count is out of range");
        children = static_cast<uint32_t>(count);
        fixed = real(required(payload, "fixed"));
        variable = real(required(payload, "variable"));
        Ref all_profiles(PySequence_Fast(required(payload, "profiles"), "Expected profiles"));
        Ref group_sizes(PySequence_Fast(required(payload, "groups"), "Expected gene group sizes"));
        Py_ssize_t n_profiles = PySequence_Fast_GET_SIZE(all_profiles.value);
        if (!n_profiles || PySequence_Fast_GET_SIZE(group_sizes.value) != n_profiles)
            invalid("Profile and gene-group dimensions disagree");
        unsigned shift = 2+(remaining_width = bits(children));
        for (Py_ssize_t p = 0; p < n_profiles; ++p) {
            Ref nodes(PySequence_Fast(PySequence_Fast_GET_ITEM(all_profiles.value, p), "Expected locus records"));
            Py_ssize_t size = PySequence_Fast_GET_SIZE(nodes.value);
            if (!size || uint64_t(size) > UINT32_MAX) invalid("Native locus catalog is empty or too large");
            std::vector<Locus> profile;
            profile.reserve(size);
            for (Py_ssize_t i = 0; i < size; ++i) {
                Ref fields(PySequence_Fast(PySequence_Fast_GET_ITEM(nodes.value, i), "Expected a locus record"));
                if (PySequence_Fast_GET_SIZE(fields.value) != 5) invalid("Native locus record has wrong size");
                Locus node;
                node.affine = triple(PySequence_Fast_GET_ITEM(fields.value, 0));
                node.risk = triple(PySequence_Fast_GET_ITEM(fields.value, 1));
                node.negative = triple(PySequence_Fast_GET_ITEM(fields.value, 2));
                Ref constant(PySequence_Fast(PySequence_Fast_GET_ITEM(fields.value, 3), "Expected support flags"));
                Ref roles(PySequence_Fast(PySequence_Fast_GET_ITEM(fields.value, 4), "Expected role transitions"));
                if (PySequence_Fast_GET_SIZE(constant.value) != 3 || PySequence_Fast_GET_SIZE(roles.value) != 3)
                    invalid("Native role dimensions disagree");
                for (int role = 0; role < 3; ++role) {
                    int flag = PyObject_IsTrue(PySequence_Fast_GET_ITEM(constant.value, role));
                    if (flag < 0) throw PythonFailure{};
                    node.constant[role] = flag;
                    Ref outcomes(PySequence_Fast(PySequence_Fast_GET_ITEM(roles.value, role), "Expected outcomes"));
                    if (PySequence_Fast_GET_SIZE(outcomes.value) > 3) invalid("Too many genotype outcomes");
                    Sum mass;
                    for (Py_ssize_t j = 0; j < PySequence_Fast_GET_SIZE(outcomes.value); ++j) {
                        Ref outcome(PySequence_Fast(PySequence_Fast_GET_ITEM(outcomes.value, j), "Expected probability and ID"));
                        if (PySequence_Fast_GET_SIZE(outcome.value) != 2) invalid("Malformed genotype transition");
                        double probability = real(PySequence_Fast_GET_ITEM(outcome.value, 0));
                        uint64_t next = integer(PySequence_Fast_GET_ITEM(outcome.value, 1));
                        if (probability <= 0 || next >= uint64_t(size)) invalid("Invalid positive transition or locus ID");
                        node.successors[role].push_back({probability, static_cast<uint32_t>(next)});
                        mass.add(probability);
                    }
                    if (!node.successors[role].empty() && std::abs(mass.total()-1.0) > 1e-12)
                        invalid("Native genotype transition probabilities do not sum to one");
                }
                profile.push_back(std::move(node));
            }
            uint64_t multiplicity = integer(PySequence_Fast_GET_ITEM(group_sizes.value, p));
            if (!multiplicity || multiplicity > PY_SSIZE_T_MAX) invalid("Invalid gene multiplicity");
            unsigned width = bits(uint64_t(size)-1);
            if (uint64_t(shift)+uint64_t(width)*multiplicity > 64)
                invalid("Exact packed state requires more than 64 bits");
            size_t begin = gene_profile.size();
            for (uint64_t j = 0; j < multiplicity; ++j) {
                gene_profile.push_back(p);
                shifts.push_back(shift);
                widths.push_back(width);
                shift += width;
            }
            spans.push_back({begin, gene_profile.size()});
            profiles.push_back(std::move(profile));
        }
        scratch.resize(gene_profile.size());
        Ref values(PySequence_Fast(root, "Expected canonical root state"));
        if (PySequence_Fast_GET_SIZE(values.value) != Py_ssize_t(gene_profile.size()+2))
            invalid("Native root dimension differs from its gene catalog");
        uint64_t mask = integer(PySequence_Fast_GET_ITEM(values.value, 0));
        uint64_t remaining = integer(PySequence_Fast_GET_ITEM(values.value, 1));
        if (mask > 3 || remaining > children) invalid("Invalid root parent flags or child count");
        std::vector<uint32_t> identifiers(gene_profile.size());
        for (size_t g = 0; g < identifiers.size(); ++g) {
            uint64_t id = integer(PySequence_Fast_GET_ITEM(values.value, g+2));
            if (id >= profiles[gene_profile[g]].size()) invalid("Root locus ID is outside its catalog");
            identifiers[g] = id;
        }
        root_key = pack(mask, remaining, identifiers);
        if (limits != Py_None) {
            Ref caps(PySequence_Fast(limits, "Expected three optional resource limits"));
            if (PySequence_Fast_GET_SIZE(caps.value) != 3) invalid("Expected three optional resource limits");
            PyObject* state_cap = PySequence_Fast_GET_ITEM(caps.value, 0);
            PyObject* transition_cap = PySequence_Fast_GET_ITEM(caps.value, 1);
            PyObject* time_cap = PySequence_Fast_GET_ITEM(caps.value, 2);
            if (state_cap != Py_None && !(max_states = integer(state_cap))) invalid("State limit must be positive");
            if (transition_cap != Py_None && !(max_transitions = integer(transition_cap))) invalid("Transition limit must be positive");
            if (time_cap != Py_None && (max_seconds = real(time_cap)) <= 0) invalid("Time limit must be positive");
        }
        if (callback != Py_None) {
            if (!PyCallable_Check(callback)) invalid("Progress callback must be callable");
            progress = callback;
            Py_INCREF(progress);
        }
    }

    uint64_t pack(unsigned mask, uint32_t remaining, const std::vector<uint32_t>& ids) {
        std::copy(ids.begin(), ids.end(), scratch.begin());
        for (auto span : spans) std::sort(scratch.begin()+span.first, scratch.begin()+span.second);
        uint64_t key = uint64_t(mask) | (uint64_t(remaining) << 2);
        for (size_t g = 0; g < ids.size(); ++g)
            if (widths[g]) key |= uint64_t(scratch[g]) << shifts[g];
        return key;
    }

    void tick(bool force = false) {
        if (!force && (++ticks & 4095)) return;
        auto now = Clock::now();
        double seconds = std::chrono::duration<double>(now-started).count();
        if (max_seconds && seconds >= max_seconds) throw LimitFailure{"max_seconds"};
        bool notify = progress && std::chrono::duration<double>(now-last_progress).count() >= 1.0;
        PyGILState_STATE gil = PyGILState_Ensure();
        bool failed = PyErr_CheckSignals() < 0;
        if (!failed && notify) {
            last_progress = now;
            PyObject* response = PyObject_CallFunction(progress, "KKKKd",
                static_cast<unsigned long long>(states), static_cast<unsigned long long>(transitions),
                static_cast<unsigned long long>(terminal), static_cast<unsigned long long>(independent_count), seconds);
            failed = response == nullptr;
            Py_XDECREF(response);
        }
        PyGILState_Release(gil);
        if (failed) throw PythonFailure{};
    }

    void panels(int role, size_t gene, double probability,
                const std::vector<const Locus*>& nodes, std::vector<uint32_t>& next,
                unsigned next_mask, uint32_t next_remaining, Sum& mass, Sum& expectation) {
        if (gene < nodes.size()) {
            const auto& outcomes = nodes[gene]->successors[role];
            if (outcomes.empty()) throw std::runtime_error("A reachable role has no genotype transitions");
            for (const auto& outcome : outcomes) {
                next[gene] = outcome.next;
                panels(role, gene+1, probability*outcome.probability, nodes, next,
                       next_mask, next_remaining, mass, expectation);
            }
            return;
        }
        tick();
        if (max_transitions && transitions >= max_transitions) throw LimitFailure{"max_transitions"};
        if (probability <= 0 || !std::isfinite(probability))
            throw std::runtime_error("Positive complete-panel probability underflowed or is nonfinite");
        ++transitions;
        uint64_t successor = pack(next_mask, next_remaining, next);
        expectation.add(probability*visit(successor));
        mass.add(probability);
    }

    double visit(uint64_t key) {
        tick();
        auto cached = memo.find(key);
        if (cached != memo.end()) return cached->second.value;
        if (max_states && states >= max_states) throw LimitFailure{"max_states"};
        ++states;
        unsigned mask = key & 3;
        uint32_t remaining = (key >> 2) & low_mask(remaining_width);
        std::array<int, 3> counts{int(!(mask & 1)), int(!(mask & 2)), int(remaining)};
        std::vector<const Locus*> nodes;
        nodes.reserve(gene_profile.size());
        std::array<Sum, 3> affine_sum, risk_sum;
        std::array<double, 3> negative{1.0, 1.0, 1.0};
        bool independent = true;
        for (size_t g = 0; g < gene_profile.size(); ++g) {
            uint32_t id = widths[g] ? ((key >> shifts[g]) & low_mask(widths[g])) : 0;
            const Locus& node = profiles[gene_profile[g]][id];
            nodes.push_back(&node);
            for (int role = 0; role < 3; ++role) {
                affine_sum[role].add(node.affine[role]);
                risk_sum[role].add(node.risk[role]);
                negative[role] *= node.negative[role];
            }
            independent = independent && (node.constant[0] || node.constant[1])
                && (!remaining || ((node.constant[0] || node.constant[2])
                                   && (node.constant[1] || node.constant[2])));
        }
        std::array<double, 3> stop, test, best;
        std::array<double, 4> scores{};
        std::array<bool, 4> present{true, false, false, false};
        Sum stopping;
        for (int role = 0; role < 3; ++role) {
            stop[role] = affine_sum[role].total()-risk_sum[role].total();
            test[role] = affine_sum[role].total()-fixed-variable*(1-negative[role]);
            best[role] = std::max(stop[role], test[role]);
            if (counts[role]) { stopping.add(counts[role]*stop[role]); present[role+1] = true; }
        }
        scores[0] = stopping.total();
        if (independent) {
            int legal = counts[0]+counts[1]+counts[2];
            terminal += legal == 1;
            independent_count += legal > 1;
            for (int chosen = 0; chosen < 3; ++chosen) if (counts[chosen]) {
                Sum value;
                value.add(test[chosen]);
                for (int role = 0; role < 3; ++role) if (counts[role])
                    value.add((counts[role]-int(role == chosen))*best[role]);
                scores[chosen+1] = value.total();
            }
        } else {
            std::vector<uint32_t> next(gene_profile.size());
            for (int role = 0; role < 3; ++role) if (counts[role]) {
                Sum mass, expectation;
                panels(role, 0, 1.0, nodes, next, role < 2 ? mask | (1u << role) : mask,
                       remaining-uint32_t(role == 2), mass, expectation);
                if (std::abs(mass.total()-1.0) > 1e-10)
                    throw std::runtime_error("Complete-panel probabilities do not sum to one");
                scores[role+1] = test[role]+expectation.total();
            }
        }
        int selected = 0;
        for (int action = 0; action < 4; ++action) if (present[action]) {
            if (!std::isfinite(scores[action])) throw std::runtime_error("Nonfinite Bellman action value");
            if (scores[action] > scores[selected]) selected = action;
        }
        memo.emplace(key, Result{scores[selected], selected-1});
        if (key == root_key) { root_scores = scores; root_present = present; }
        return scores[selected];
    }
};

struct MemoIterator {
    PyObject_HEAD
    PyObject* owner;
    Memo::const_iterator* cursor;
};
PyTypeObject memo_iterator_type = {PyVarObject_HEAD_INIT(nullptr, 0)};

void destroy_iterator(PyObject* object) {
    auto* iterator = reinterpret_cast<MemoIterator*>(object);
    delete iterator->cursor;
    Py_XDECREF(iterator->owner);
    Py_TYPE(object)->tp_free(object);
}
PyObject* next_key(PyObject* object) {
    auto* iterator = reinterpret_cast<MemoIterator*>(object);
    auto* state = static_cast<State*>(PyCapsule_GetPointer(iterator->owner, capsule_name));
    if (!state) return nullptr;
    if (*iterator->cursor == state->memo.cend()) return nullptr;
    uint64_t key = (*iterator->cursor)->first;
    ++*iterator->cursor;
    return PyLong_FromUnsignedLongLong(key);
}
PyObject* keys(PyObject*, PyObject* capsule) {
    auto* state = static_cast<State*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (!state) return nullptr;
    auto* iterator = PyObject_New(MemoIterator, &memo_iterator_type);
    if (!iterator) return nullptr;
    iterator->owner = capsule;
    iterator->cursor = nullptr;
    Py_INCREF(capsule);
    try { iterator->cursor = new Memo::const_iterator(state->memo.cbegin()); }
    catch (const std::bad_alloc&) { Py_DECREF(iterator); return PyErr_NoMemory(); }
    return reinterpret_cast<PyObject*>(iterator);
}
PyObject* size(PyObject*, PyObject* capsule) {
    auto* state = static_cast<State*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (!state) return nullptr;
    return PyLong_FromSize_t(state->memo.size());
}

void destroy_capsule(PyObject* capsule) {
    auto* state = static_cast<State*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (state) delete state;
    else PyErr_Clear();
}
void set_item(PyObject* mapping, const char* name, PyObject* value) {
    Ref owned(value);
    if (PyDict_SetItemString(mapping, name, value) < 0) throw PythonFailure{};
}

PyObject* solve(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject *payload, *root, *limits = Py_None, *progress = Py_None;
    static const char* names[] = {"payload", "root", "limits", "progress", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OO", const_cast<char**>(names),
                                    &payload, &root, &limits, &progress)) return nullptr;
    try {
        auto state = std::make_unique<State>();
        state->parse(payload, root, limits, progress);
        std::string limit_reason, error_message;
        bool python_error = false, no_memory = false;
        PyThreadState* thread = PyEval_SaveThread();
        try {
            state->tick(true);
            state->visit(state->root_key);
            state->tick(true);
        } catch (const LimitFailure& error) { limit_reason = error.reason; }
          catch (const PythonFailure&) { python_error = true; }
          catch (const std::bad_alloc&) { no_memory = true; }
          catch (const std::exception& error) { error_message = error.what(); }
        PyEval_RestoreThread(thread);
        if (python_error) return nullptr;
        if (no_memory) return PyErr_NoMemory();
        if (!limit_reason.empty()) {
            Ref details(Py_BuildValue("(sKKd)", limit_reason.c_str(),
                static_cast<unsigned long long>(state->states),
                static_cast<unsigned long long>(state->transitions), state->elapsed()));
            PyErr_SetObject(limit_exception, details.value);
            return nullptr;
        }
        if (!error_message.empty()) { PyErr_SetString(PyExc_RuntimeError, error_message.c_str()); return nullptr; }
        Py_CLEAR(state->progress);
        Ref output(PyDict_New());
        Ref scores(PyDict_New());
        for (int index = 0; index < 4; ++index) if (state->root_present[index]) {
            Ref action(PyLong_FromLong(index-1));
            Ref value(PyFloat_FromDouble(state->root_scores[index]));
            if (PyDict_SetItem(scores.value, action.value, value.value) < 0) throw PythonFailure{};
        }
        if (PyDict_SetItemString(output.value, "root_role_values", scores.value) < 0) throw PythonFailure{};
        set_item(output.value, "root_value", PyFloat_FromDouble(state->memo.at(state->root_key).value));
        set_item(output.value, "states_evaluated", PyLong_FromUnsignedLongLong(state->states));
        set_item(output.value, "transitions_evaluated", PyLong_FromUnsignedLongLong(state->transitions));
        set_item(output.value, "terminal_actions_closed", PyLong_FromUnsignedLongLong(state->terminal));
        set_item(output.value, "independent_states_closed", PyLong_FromUnsignedLongLong(state->independent_count));
        set_item(output.value, "native_seconds", PyFloat_FromDouble(state->elapsed()));
        Ref capsule(PyCapsule_New(state.get(), capsule_name, destroy_capsule));
        if (PyDict_SetItemString(output.value, "capsule", capsule.value) < 0) {
            PyCapsule_SetDestructor(capsule.value, nullptr);
            throw PythonFailure{};
        }
        state.release();
        Py_INCREF(output.value);
        return output.value;
    } catch (const PythonFailure&) { return nullptr; }
      catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
      catch (const std::exception& error) { PyErr_SetString(PyExc_RuntimeError, error.what()); return nullptr; }
}

PyObject* lookup(PyObject*, PyObject* args) {
    PyObject* capsule;
    unsigned long long key;
    if (!PyArg_ParseTuple(args, "OK", &capsule, &key)) return nullptr;
    auto* state = static_cast<State*>(PyCapsule_GetPointer(capsule, capsule_name));
    if (!state) return nullptr;
    auto found = state->memo.find(key);
    if (found == state->memo.end()) Py_RETURN_NONE;
    return Py_BuildValue("(di)", found->second.value, found->second.role);
}

PyMethodDef methods[] = {
    {"solve", reinterpret_cast<PyCFunction>(solve), METH_VARARGS | METH_KEYWORDS, "Solve the full finite Bellman problem."},
    {"lookup", lookup, METH_VARARGS, "Read a completed canonical state value and role."},
    {"keys", keys, METH_O, "Iterate completed packed state keys without copying the map."},
    {"size", size, METH_O, "Return the number of completed cached states."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", nullptr, -1, methods};
}

PyMODINIT_FUNC PyInit__native() {
    memo_iterator_type.tp_name = "pedigree_panel_scaling._native.MemoIterator";
    memo_iterator_type.tp_basicsize = sizeof(MemoIterator);
    memo_iterator_type.tp_flags = Py_TPFLAGS_DEFAULT;
    memo_iterator_type.tp_dealloc = destroy_iterator;
    memo_iterator_type.tp_iter = PyObject_SelfIter;
    memo_iterator_type.tp_iternext = next_key;
    if (PyType_Ready(&memo_iterator_type) < 0) return nullptr;
    PyObject* result = PyModule_Create(&module);
    if (!result) return nullptr;
    limit_exception = PyErr_NewException("_native.LimitExceeded", PyExc_RuntimeError, nullptr);
    if (!limit_exception || PyModule_AddObject(result, "LimitExceeded", limit_exception) < 0) {
        Py_XDECREF(limit_exception);
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}
