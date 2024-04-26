def get_state_names(num_states, accepting_state, rejecting_state):
    states = ["u" + str(i) for i in range(num_states - 2)]
    if accepting_state is not None:
        states.append(accepting_state)
    if rejecting_state is not None:
        states.append(rejecting_state)
    return states


def generate_state_statements(num_states, accepting_state, rejecting_state):
    states = get_state_names(num_states, accepting_state, rejecting_state)
    return "".join(["state(" + s + ").\n" for s in states]) + '\n'


def generate_state_id_statements(num_states):
    return "".join([f"state_id(u{i}, {i}).\n" for i in range(num_states - 2)])


def generate_state_id_restrictions():
    # A state appears if it is the source or destination state on an edge and has an id.
    stmt = "state_appears(X) :- ed(X, _, _), state_id(X, _)."
    stmt += "state_appears(X) :- ed(_, X, _), state_id(X, _)."

    # The hypothesis is not OK if a state id lower than one appearing in the RM is unused.
    stmt += ":- state_id(X, XID), state_id(Y, YID), state_appears(X), not state_appears(Y), YID < XID."
    return stmt


def get_state_priorities(num_states, accepting_state, rejecting_state, use_terminal_priority):
    states = get_state_names(num_states, accepting_state, rejecting_state)
    states_with_priority = states if use_terminal_priority \
        else [s for s in states if s != accepting_state and s != rejecting_state]
    state_priorities = []
    for s, priority in zip(states_with_priority, range(len(states_with_priority))):
        state_priorities.append((s, priority))
    return state_priorities
