#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#

import logging

import numpy as np
from pyboy.utils import WindowEvent
from collections import namedtuple

from .base_plugin import PyBoyGameWrapper

logger = logging.getLogger(__name__)

try:
    from cython import compiled
    cythonmode = compiled
except ImportError:
    cythonmode = False

#enumeration of the different scenes available in the game
scenes = {"overworld": 0, "wild": 1, "trainer": 2, "gym": 3, "elite_four": 0}

Pokemon = namedtuple('Pokemon', ['hp', 'max_hp', 'level', 'type']) #simplified Pokemon representation

#Reads and returns BCM value of 2 bytes big-endian memory value.
def read_big_endian(value1, value2):
    return (value1 << 8) + value2


class GameWrapperPokemonGold(PyBoyGameWrapper):
    """
    This class wraps Pokémon Gold, and provides easy access to score, coins, lives left, time left, world and a
    "fitness" score for AIs.

    __Only world 1-1 is officially supported at the moment. Support for more worlds coming soon.__

    If you call `print` on an instance of this object, it will show an overview of everything this object provides.
    """
    cartridge_title = "POKEMON GOLD"

    def __init__(self, *args, **kwargs):
        self.shape = (20, 16)
        """The shape of the game area"""
        self.badges = 0
        """Provides the current number of badges."""
        self.money = 0
        """The amount of money."""
        self.party_size = 0
        """The number of pokemons in party"""
        self.low_hp = False
        """This value is set to True if the active Pokemon has low hp."""
        self.scene = 0

        #Party state
        self.current_poke_hp = 0
        """HP of the first pokemon of the party (current pokemon)."""
        self.current_poke_max_hp = 0
        """Max HP of the first pokemon of the party (current pokemon)."""

        #Battle state
        self.opponent_poke_HP = 0
        """HP of the current opponent Pokemon."""

        self.fitness = 0
        """
        A built-in fitness scoring. Taking points, level progression, time left, and lives left into account.

        .. math::
            fitness = (lives\\_left \\cdot 10000) + (score + time\\_left \\cdot 10) + (\\_level\\_progress\\_max \\cdot 10)
        """

        super().__init__(*args, game_area_section=(0, 2) + self.shape, game_area_wrap_around=True, **kwargs)

    def post_tick(self):
        self._tile_cache_invalid = True
        self._sprite_cache_invalid = True
        
        self.update_badges()
        self.update_scene()
        self.update_current_poke_hp()

        if self.scene != "overworld":
            self.update_battle_stats()

    def update_current_poke_hp(self):
        val = _bcm_to_dec(self.pyboy.get_memory_value(0xC1A6))
        if val > 80:
            self.low_hp = True
        else:
            self.low_hp = False
        self.current_poke_hp = read_big_endian(self.pyboy.get_memory_value(0xDA4C), self.pyboy.get_memory_value(0xDA4D))
        self.current_poke_max_hp = read_big_endian(self.pyboy.get_memory_value(0xDA4E), self.pyboy.get_memory_value(0xDA4F))


    def update_battle_stats(self):
        self.opponent_poke_HP = read_big_endian(self.pyboy.get_memory_value(0xD0FF), self.pyboy.get_memory_value(0xD100))    
    
    def update_badges(self):
        self.badges = self.pyboy.get_memory_value(0xD57C)
    
    def update_scene(self):
        self.scene = list(scenes.keys())[list(scenes.values()).index(self.pyboy.get_memory_value(0xD116))]
    
    
    def get_elite_four(self):
        """
        get elite four
        """
        pass

    def start_game(self, timer_div=None, world_level=None, unlock_level_select=False):
        """
        Call this function right after initializing PyBoy. This will start a game in world 1-1 and give back control on
        the first frame it's possible.

        The state of the emulator is saved, and using `reset_game`, you can get back to this point of the game
        instantly.

        The game has 4 major worlds with each 3 level. to start at a specific world and level, provide it as a tuple for
        the optional keyword-argument `world_level`.

        If you're not using the game wrapper for unattended use, you can unlock the level selector for the main menu.
        Enabling the selector, will make this function return before entering the game.

        Kwargs:
            timer_div (int): Replace timer's DIV register with this value. Use `None` to randomize.
            world_level (tuple): (world, level) to start the game from
            unlock_level_select (bool): Unlock level selector menu
        """
        PyBoyGameWrapper.start_game(self, timer_div=timer_div)

        if world_level is not None:
            self.set_world_level(*world_level)

        # Boot screen
        while True:
            self.pyboy.tick()
            if self.tilemap_background[6:11, 13] == [284, 285, 266, 283, 285]: # "START" on the main menu
                break
        self.pyboy.tick()
        self.pyboy.tick()
        self.pyboy.tick()

        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

        while True:
            if unlock_level_select and self.pyboy.frame_count == 71: # An arbitrary frame count, where the write will work
                self.pyboy.set_memory_value(ADDR_WIN_COUNT, 2 if unlock_level_select else 0)
                break
            self.pyboy.tick()
            self.tilemap_background.refresh_lcdc()

            # "MARIO" in the title bar and 0 is placed at score
            if self.tilemap_background[0:5, 0] == [278, 266, 283, 274, 280] and \
               self.tilemap_background[5, 1] == 256:
                self.game_has_started = True
                break

        self.saved_state.seek(0)
        self.pyboy.save_state(self.saved_state)

        self._set_timer_div(timer_div)

    def reset_game(self, timer_div=None):
        """
        After calling `start_game`, use this method to reset Mario to the beginning of world 1-1.

        If you want to reset to later parts of the game -- for example world 1-2 or 3-1 -- use the methods
        `pyboy.PyBoy.save_state` and `pyboy.PyBoy.load_state`.

        Kwargs:
            timer_div (int): Replace timer's DIV register with this value. Use `None` to randomize.
        """
        PyBoyGameWrapper.reset_game(self, timer_div=timer_div)

        self._set_timer_div(timer_div)

    def game_area(self):
        """
        Use this method to get a matrix of the "game area" of the screen. This view is simplified to be perfect for
        machine learning applications.

        In Super Mario Land, this is almost the entire screen, expect for the top part showing the score, lives left
        and so on. These values can be found in the variables of this class.

        In this example, Mario is `0`, `1`, `16` and `17`. He is standing on the ground which is `352` and `353`:
        ```text
             0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19
        ____________________________________________________________________________________
        0  | 339 339 339 339 339 339 339 339 339 339 339 339 339 339 339 339 339 339 339 339
        1  | 320 320 320 320 320 320 320 320 320 320 320 320 320 320 320 320 320 320 320 320
        2  | 300 300 300 300 300 300 300 300 300 300 300 300 321 322 321 322 323 300 300 300
        3  | 300 300 300 300 300 300 300 300 300 300 300 324 325 326 325 326 327 300 300 300
        4  | 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300
        5  | 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300
        6  | 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300 300
        7  | 300 300 300 300 300 300 300 300 310 350 300 300 300 300 300 300 300 300 300 300
        8  | 300 300 300 300 300 300 300 310 300 300 350 300 300 300 300 300 300 300 300 300
        9  | 300 300 300 300 300 129 310 300 300 300 300 350 300 300 300 300 300 300 300 300
        10 | 300 300 300 300 300 310 300 300 300 300 300 300 350 300 300 300 300 300 300 300
        11 | 300 300 310 350 310 300 300 300 300 306 307 300 300 350 300 300 300 300 300 300
        12 | 300 368 369 300 0   1   300 306 307 305 300 300 300 300 350 300 300 300 300 300
        13 | 310 370 371 300 16  17  300 305 300 305 300 300 300 300 300 350 300 300 300 300
        14 | 352 352 352 352 352 352 352 352 352 352 352 352 352 352 352 352 352 352 352 352
        15 | 353 353 353 353 353 353 353 353 353 353 353 353 353 353 353 353 353 353 353 353
        ```

        Returns
        -------
        memoryview:
            Simplified 2-dimensional memoryview of the screen
        """
        return PyBoyGameWrapper.game_area(self)

    def game_over(self):
        # Apparantly that address is for game over
        # https://datacrystal.romhacking.net/wiki/Super_Mario_Land:RAM_map
        return self.pyboy.get_memory_value(0xC0A4) == 0x39

    def __repr__(self):
        adjust = 4
        # yapf: disable
        return (
            f"Super Mario Land: World {'-'.join([str(i) for i in self.world])}\n" +
            f"Coins: {self.coins}\n" +
            f"lives_left: {self.lives_left}\n" +
            f"Score: {self.score}\n" +
            f"Time left: {self.time_left}\n" +
            f"Level progress: {self.level_progress}\n" +
            f"Fitness: {self.fitness}\n" +
            "Sprites on screen:\n" +
            "\n".join([str(s) for s in self._sprites_on_screen()]) +
            "\n" +
            "Tiles on screen:\n" +
            " "*5 + "".join([f"{i: <4}" for i in range(20)]) + "\n" +
            "_"*(adjust*20+4) +
            "\n" +
            "\n".join(
                [
                    f"{i: <3}| " + "".join([str(tile).ljust(adjust) for tile in line])
                    for i, line in enumerate(self.game_area())
                ]
            )
        )
        # yapf: enable
