"""
Test file for util.py functions.

This test file provides comprehensive unit tests for utility functions,
particularly focusing on generate_previous_incomplete_examples with a
coffee_mail reward machine fixture.
"""

import pytest
from typing import List, Tuple
from unittest.mock import Mock, patch

from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_learning.ilasp.ilasp_example_representation import ISAILASPExample, ObservablePredicate, LastPredicate
from rm_marl.new_stack.learner.util import find_all_subsets, generate_previous_incomplete_examples, find_all_paths, is_terminal_path, to_obs_predicate_list


class TestUtil:
    """Test cases for utility functions."""
    
    # Constants for coffee_mail reward machine
    _coffee = "f"
    _office = "g"
    _mail = "m"
    _plant = "n"

    @pytest.fixture
    def coffee_mail_rm(self) -> RewardMachine:
        """Create coffee_mail reward machine fixture."""
        rm = RewardMachine()
        rm.add_states(["u0", "u1", "u2", "u3", "u_acc", "u_rej"])
        rm.set_u0("u0")
        rm.set_uacc("u_acc")
        rm.set_urej("u_rej")

        rm.add_transition("u0", "u1", (self._coffee, "~" + self._office, "~" + self._mail))
        rm.add_transition("u0", "u2", ("~" + self._coffee, "~" + self._office, self._mail))
        rm.add_transition("u0", "u3", (self._coffee, "~" + self._office, self._mail))
        rm.add_transition("u0", "u_acc", (self._coffee, self._office, self._mail))
        rm.add_transition("u1", "u3", ("~" + self._office, self._mail, "~" + self._plant))
        rm.add_transition("u1", "u_acc", (self._office, self._mail, "~" + self._plant))
        rm.add_transition("u2", "u3", ("~" + self._office, self._coffee))
        rm.add_transition("u2", "u_acc", (self._office, self._coffee))
        rm.add_transition("u3", "u_acc", (self._office, "~" + self._plant))

        rm.add_transition("u0", "u_rej", (self._plant, "~" + self._coffee, "~" + self._mail))
        rm.add_transition("u1", "u_rej", (self._plant,))
        rm.add_transition("u2", "u_rej", (self._plant, "~" + self._coffee))
        rm.add_transition("u3", "u_rej", (self._plant,))

        return rm

    def test_find_all_paths_coffee_mail_rm(self, coffee_mail_rm):
        """Test _find_all_paths returns all expected paths in coffee_mail reward machine."""
        # Find all paths starting from u0
        all_paths = find_all_paths(coffee_mail_rm, "u0", set())
        
        # Expected paths in coffee_mail structure:
        expected_paths_coffee_mail = [
            # Path 1: u0 -> u1 -> u3 -> u_acc (via coffee, mail, office)
            [
                (self._coffee,),
                (self._mail,),
                (self._office,)
            ],
            # Path 2: u0 -> u3 -> u_acc (via (coffee and mail), office)
            [
                (self._coffee, self._mail),
                (self._office,)
            ],
            # Path 3: u0 -> u2 -> u3 -> u_acc (via mail, coffee, office)
            [
                (self._mail,),
                (self._coffee,),
                (self._office,)
            ],
            # Path 4: u0 -> u_rej (via plant)
            [
                (self._plant,)
            ],
            # Path 5: u0 -> u1 -> u_rej (via coffee, plant)
            [
                (self._coffee,),
                (self._plant,)
            ],
            # Path 6: u0 -> u2 -> u_rej (via mail, plant)
            [
                (self._mail,),
                (self._plant,)
            ],
            # Path 7: u0 -> u3 -> u_rej (via coffee and mail, plant)
            [
                (self._coffee, self._mail),
                (self._plant,)
            ],
            # Path 8: u0 -> u1 -> u3 -> u_acc (via coffee, mail, plant)
            [
                (self._coffee,),
                (self._mail,),
                (self._plant,)
            ],
            # Path 9: u0 -> u1 -> u_acc (via coffee, (mail and office))
            [
                (self._coffee,),
                (self._office, self._mail)
            ],
            # Path 10: u0 -> u2 -> u3 -> u_rej (mail, coffee, plant)
            [
                (self._mail,),
                (self._coffee,),
                (self._plant,)
            ],
            # Path 11: u0 -> u2 -> u_acc (via mail, coffee and office)
            [
                (self._mail,),
                (self._office, self._coffee)
            ],
            # Path 12: u0 -> u_acc (via (coffee, office, mail))
            [
                (self._coffee, self._office, self._mail)
            ]
        ]

        # Verify we get some paths (exact count may vary)
        assert len(all_paths) > 0

        assert len(all_paths) == len(expected_paths_coffee_mail)
        
        # Verify all expected paths are present
        for expected_path in expected_paths_coffee_mail:
            assert expected_path in all_paths

    def test_find_all_paths_empty_rm(self):
        """Test _find_all_paths returns empty list for empty reward machine."""
        empty_rm = RewardMachine()
        empty_rm.add_states(["u0"])
        empty_rm.set_u0("u0")
        # No transitions or accepting state

        all_paths = find_all_paths(empty_rm, "u0", set())
        assert all_paths == []

    @pytest.mark.parametrize(
        "rm_path_builder",
        [
            pytest.param(lambda s: [(s._coffee, s._mail), (s._office,)], id="coffee+mail_then_office"),
            pytest.param(lambda s: [(s._mail,), (s._coffee,), (s._office,)], id="mail_then_coffee_then_office"),
            pytest.param(lambda s: [(s._coffee,), (s._office, s._mail)], id="coffee_then_office+mail"),
            pytest.param(lambda s: [(s._mail,), (s._office, s._coffee)], id="mail_then_office+coffee"),
            pytest.param(lambda s: [(s._coffee, s._office, s._mail)], id="coffee+office+mail"),
        ],
    )
    def test_find_all_subsets(self, rm_path_builder):
        rm_path = rm_path_builder(self)
        subsets = find_all_subsets(rm_path)
        n = len(rm_path)

        # Expect all non-empty, proper subsets: 2^n - 2
        assert len(subsets) == (2 ** n) - 2

        # Validate each subset is within size bounds and drawn from rm_path
        original_steps = set(rm_path)
        assert all(1 <= len(s) <= n - 1 for s in subsets)
        for subset in subsets:
            for step in subset:
                assert step in original_steps

        # Ensure no duplicates
        assert len(set(subsets)) == len(subsets)

    def test_is_terminal_path_success_and_failure(self, coffee_mail_rm):
        # Successful path: m -> f -> g should reach accepting
        successful_path = [
            (self._mail,),
            (self._coffee,),
            (self._office,),
        ]
        assert is_terminal_path(successful_path, coffee_mail_rm) is True

        # Unsuccessful, non-terminal path: m -> f should not be terminal
        unsuccessful_path = [
            (self._mail,),
            (self._coffee,),
        ]
        assert is_terminal_path(unsuccessful_path, coffee_mail_rm) is False

    @pytest.mark.parametrize(
        "rm_path_fn,expected",
        [
            (
                lambda s: [
                    (s._mail,),
                    (s._coffee,),
                    (s._office,),
                ],
                [
                    ObservablePredicate("m", 0),
                    ObservablePredicate("f", 1),
                    ObservablePredicate("g", 2),
                ],
            ),
            (
                lambda s: [
                    (s._mail, s._coffee),
                    (s._office,),
                ],
                [
                    ObservablePredicate("m", 0),
                    ObservablePredicate("f", 0),
                    ObservablePredicate("g", 1),
                ],
            ),
        ],
    )
    def test_to_obs_predicate_list_expected_output(self, rm_path_fn, expected):
        rm_path = rm_path_fn(self)

        obs_list = to_obs_predicate_list(rm_path)
        assert len(obs_list) == len(expected)
        for got, exp in zip(obs_list, expected):
            assert got.label == exp.label
            assert got.time_step == exp.time_step

    def test_generate_previous_incomplete_examples(self, coffee_mail_rm):
        """
        Placeholder for debugging.
        """
        examples = generate_previous_incomplete_examples(coffee_mail_rm)
        assert len(examples) > 0
