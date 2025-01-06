import numpy as np
from typing import Tuple


class GridWorld(object):
    def __init__(self):
        self.grid_size: Tuple[int, int] = (3, 3)
        self.num_actions: int = 4  
        self.rewards: np.ndarray = np.array([[0, 0, 0],
                                             [0, 0, 0],
                                             [0, 1, 0]]
                                            )

    def get_reward(self, state: Tuple[int, int]) -> np.ndarray:
        return self.rewards[state[0], state[1]]


class ValueFunction(object):
    def __init__(self, grid_size: Tuple[int, int]):
        self.values: np.ndarray = np.zeros(grid_size)

    def update_value(self, state: Tuple[int, int], new_value: np.ndarray) -> None:
        self.values[state[0], state[1]] = new_value
        return None

    def get_value(self, state: Tuple[int, int]):
        return self.values[state[0], state[1]]


def main() -> None:
    grid_world = GridWorld()
    value_fn = ValueFunction(grid_world.grid_size)

    for i in range(grid_world.grid_size[0]):
        for j in range(grid_world.grid_size[1]):
            state: Tuple[int, int] = (i, j)
            value_fn.update_value(state, grid_world.get_reward(state))

    print("Initial value : ", value_fn.values)
    return None


if __name__ == '__main__':
    main()