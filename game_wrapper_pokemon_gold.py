#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#

import logging

import numpy as np
from collections import namedtuple

from .base_plugin import PyBoyGameWrapper

logger = logging.getLogger(__name__)

try:
    from cython import compiled
    cythonmode = compiled
except ImportError:
    cythonmode = False

#enumeration of the different scenes available in the game
scenes = {"overworld": 0, "wild": 1, "trainer": 2, "gym": 3, "elite_four": 4}

#saved_state = "/home/alexandre/Documents/perso/emerald_deep_rl/games/after_rival_pokemon_gold"

badges = [0, 0x01, 0x01+0x02, 0x01+0x02+0x04, 0x01+0x02+0x04+0x08, 0x01+0x02+0x04+0x08+0x10, 0x01+0x02+0x04+0x08+0x10+0x20, 0x01+0x02+0x04+0x08+0x10+0x20+0x40, 0xFF]
#Reads and returns 2 bytes big-endian memory value.
def read_big_endian(value1, value2):
    return (value1 << 8) + value2


class GameWrapperPokemonGold(PyBoyGameWrapper):
    """
    This class wraps Pokémon Gold, and provides easy access to score, coins, lives left, time left, world and a
    "fitness" score for AIs.

    __Only world 1-1 is officially supported at the moment. Support for more worlds coming soon.__

    If you call `print` on an instance of this object, it will show an overview of everything this object provides.
    """
    cartridge_title = "POKEMON_GLDAAU"

    def __init__(self, *args, **kwargs):
        self.shape = (20, 16)
        """The shape of the game area"""
        self.badges = 0
        """Provides the current number of badges."""
        self.money = 0
        """The amount of money."""
        self.low_hp = 0
        """This value is set to True if the active Pokemon has low hp."""
        self.scene = "overworld"
        """The current game scene."""
        self.player_location = (0, 0)
        """The current location of the player on the game map."""
        self.textbox = 0
        """This value is set to True if the player is talking to a NPC."""

        #Party state
        self.current_poke_hp = 0
        self.current_poke_max_hp = 0
        self.current_poke_level = 0
        """State of the first pokemon of the party (current pokemon)."""

        #Battle state
        self.opponent_poke_hp = 0
        self.opponent_poke_max_hp = 0
        self.opponent_poke_level = 0
        """State of the current opponent Pokemon."""

        self.fitness = 0
        """Fitness score. Computed as badges*100 + (current_hp/hp_max)*10"""

        super().__init__(*args, game_area_section=(0, 2) + self.shape, game_area_wrap_around=True, **kwargs)

    def post_tick(self):
        self._tile_cache_invalid = True
        self._sprite_cache_invalid = True
        
        self.update_badges()
        self.update_scene()
        self.update_current_poke()

        if self.scene != "overworld":
            self.update_battle_stats()
        else:
            self.update_player_location()
            self.update_textbox()
        
        self.fitness = self.badges*100 +int((self.current_poke_hp/self.current_poke_max_hp)*10)

    def update_current_poke(self):
        val = self.pyboy.get_memory_value(0xC1A6)
        if val > 80:
            self.low_hp = 1
        else:
            self.low_hp = 0
        self.current_poke_hp = read_big_endian(self.pyboy.get_memory_value(0xDA4C), self.pyboy.get_memory_value(0xDA4D))
        self.current_poke_max_hp = read_big_endian(self.pyboy.get_memory_value(0xDA4E), self.pyboy.get_memory_value(0xDA4F))
        self.current_poke_level = self.pyboy.get_memory_value(0xDA49)

    def update_battle_stats(self):
        self.opponent_poke_hp = read_big_endian(self.pyboy.get_memory_value(0xD0FF), self.pyboy.get_memory_value(0xD100))    
        self.opponent_poke_max_hp = read_big_endian(self.pyboy.get_memory_value(0xD101), self.pyboy.get_memory_value(0xD102))
        self.opponent_poke_level = self.pyboy.get_memory_value(0xD0FC)

    def update_player_location(self):
        self.player_location = (self.pyboy.get_memory_value(0xD20D), self.pyboy.get_memory_value(0xD20E))

    def update_badges(self):
        b = self.pyboy.get_memory_value(0xD57C)
        self.badges = badges.index(b)
    
    def update_scene(self):
        self.scene = list(scenes.keys())[list(scenes.values()).index(self.pyboy.get_memory_value(0xD116))]
    
    def update_money(self):
        self.money = self.pyboy.get_memory_value(0xD573) + self.pyboy.get_memory_value(0xD574) + self.pyboy.get_memory_value(0xD575)
    
    def update_textbox(self):
        self.textbox = 1 if self.pyboy.get_memory_value(0xD15B) == 4 else 0
    
    def get_elite_four(self):
        """
        get elite four
        """
        pass

    def start_game(self, timer_div=None):
        """
        Call this function right after initializing PyBoy. This will start a game by loading the saved state at the given path.

        The state of the emulator is saved, and using `reset_game`, you can get back to this point of the game
        instantly.

        Kwargs:
            timer_div (int): Replace timer's DIV register with this value. Use `None` to randomize.
            saved_stated (str): Path to the save state file.
        """
        PyBoyGameWrapper.start_game(self, timer_div=timer_div)

        # # Boot screen
        # while True:
        #     self.pyboy.tick()
        #     if self.tilemap_background[6:11, 13] == [284, 285, 266, 283, 285]: # "START" on the main menu
        #         break
        # self.pyboy.tick()
        # self.pyboy.tick()
        # self.pyboy.tick()

        # self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        # self.pyboy.tick()
        # self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

        # while True:
        #     if unlock_level_select and self.pyboy.frame_count == 71: # An arbitrary frame count, where the write will work
        #         self.pyboy.set_memory_value(ADDR_WIN_COUNT, 2 if unlock_level_select else 0)
        #         break
        #     self.pyboy.tick()
        #     self.tilemap_background.refresh_lcdc()

        #     # "MARIO" in the title bar and 0 is placed at score
        #     if self.tilemap_background[0:5, 0] == [278, 266, 283, 274, 280] and \
        #        self.tilemap_background[5, 1] == 256:
        #         self.game_has_started = True
        #         break

        # self.saved_state.seek(0)
        # self.pyboy.save_state(self.saved_state)

        self.game_has_started = True
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
        return self.current_poke_hp == 0

    def __repr__(self):
        # yapf: disable
        return (
            f"Pokemon Gold\n" +
            f"Player Location in current map: ({self.player_location})\n" +
            f"Number of badges: {self.badges}\n" +
            f"Money: {self.money} Pokedollars\n"+
            f"Current Poke HP: {self.current_poke_hp}/{self.current_poke_max_hp}\n" +
            f"Current Poke Level: {self.current_poke_level}\n" +
            f"Scene: {self.scene}\n" +
            f"Opponent Poke HP: {'NA' if self.opponent_poke_hp == -1 else self.opponent_poke_hp}/{'NA' if self.opponent_poke_max_hp == -1 else self.opponent_poke_max_hp}\n" +
            f"Opponent Poke Level: {'NA' if self.opponent_poke_level == -1 else self.opponent_poke_level}\n" +
            f"Fitness: {self.fitness}"
            )
        # yapf: enable
