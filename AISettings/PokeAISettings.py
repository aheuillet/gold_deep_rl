from asyncio import sleep
import itertools
from pyboy import WindowEvent
from AISettings.AISettingsInterface import AISettingsInterface, Config

# Macros
DMG_REWARD = 10
FAINT_REWARD = 100
ENCOUNTER_REWARD = 1000
GYM_REWARD = 10000

class GameState():
    def __init__(self, pyboy):
        game_wrapper = pyboy.game_wrapper()
        self.money = game_wrapper.money
        self.badges = game_wrapper.badges
        self.current_poke_hp = game_wrapper.current_poke_hp
        self.current_poke_max_hp = game_wrapper.current_poke_max_hp
        self.current_poke_level = game_wrapper.current_poke_level
        self.opponent_poke_hp = game_wrapper.opponent_poke_hp
        self.opponent_poke_max_hp = game_wrapper.opponent_poke_max_hp
        self.opponent_poke_level = game_wrapper.opponent_poke_level
        self.scene = game_wrapper.scene
        self.low_hp = bool(game_wrapper.low_hp)
        self.player_location = game_wrapper.player_location
        self.textbox = bool(game_wrapper.textbox)

class PokeAI(AISettingsInterface):
	def __init__(self):
		self.realMax = [] #[[1,1, 2500], [1,1, 200]]		

	def GetReward(self, prevGameState: GameState, pyboy):
		# timeRespawn = pyboy.get_memory_value(0xFFA6) #Time until respawn from death (set when Mario has fell to the bottom of the screen) 
		# if(timeRespawn > 0): # if we cant move return 0 reward otherwise we could be punished for crossing a level
		# 	return 0

		"Get current game state"
		current_state = self.GetGameState(pyboy)

		reward = 0

		if current_state.scene == "overworld":
			#reward += self.ComputeMovementReward(prevGameState, current_state)
			reward += self.ComputeBadgeReward(prevGameState, current_state)
		elif current_state.scene == "wild":
			if prevGameState.scene == "overworld":
				reward += 0.1
			reward += self.ComputeBattleReward(prevGameState, current_state)
		elif current_state.scene == "trainer":
			if prevGameState.scene == "overworld":
				reward += 0.5
			reward += self.ComputeBattleReward(prevGameState, current_state)
		elif current_state.scene == "gym":
			if prevGameState.scene == "overworld":
				reward += 1
			reward += self.ComputeBattleReward(prevGameState, current_state)

		return reward
	
	def ComputeBattleReward(self, prevGameState: GameState, currentGameState: GameState):
		poke_hp_diff = currentGameState.current_poke_hp - prevGameState.current_poke_hp
		opponent_hp_diff = currentGameState.opponent_poke_hp - prevGameState.opponent_poke_hp
		if opponent_hp_diff < 0:
			opponent_reward = - DMG_REWARD*(opponent_hp_diff/currentGameState.opponent_poke_max_hp)
		else:
			opponent_reward = 0
		low_hp_reward = -0.05 if currentGameState.low_hp else 0
		return DMG_REWARD*(poke_hp_diff/currentGameState.current_poke_max_hp) + opponent_reward + low_hp_reward
	
	def ComputeBadgeReward(self, prevGameState: GameState, currentGameState: GameState):
		return 1 if currentGameState.badges > prevGameState.badges else 0

	def ComputeMovementReward(self, prevGameState: GameState, currentGameState: GameState):
		if prevGameState.player_location == currentGameState.player_location: #and currentGameState.textbox == False:
			return -0.0001
		else:
			return 0

	def GetActions(self):
		baseActions = [WindowEvent.PRESS_BUTTON_A,
                       WindowEvent.PRESS_BUTTON_B,
                       WindowEvent.PRESS_ARROW_UP,
                       WindowEvent.PRESS_ARROW_DOWN,
                       WindowEvent.PRESS_ARROW_LEFT,
                       WindowEvent.PRESS_ARROW_RIGHT,
                       WindowEvent.PRESS_BUTTON_START
                       ]

		totalActionsWithRepeats = list(itertools.permutations(baseActions, 2))
		withoutRepeats = []

		# for combination in totalActionsWithRepeats:
		# 	reversedCombination = combination[::-1]
		# 	if(reversedCombination not in withoutRepeats):
		# 		withoutRepeats.append(combination)

		filteredActions = [[action] for action in baseActions] + withoutRepeats

		return filteredActions

	def PrintGameState(self, pyboy):
		#gameState = GameState(pyboy)
		game_wrapper = pyboy.game_wrapper()

		print(repr(game_wrapper))

	def GetGameState(self, pyboy):
		return GameState(pyboy)

	def GetHyperParameters(self) -> Config:
		config = Config()
		config.exploration_rate_decay = 0.999
		return config

	def GetLength(self, pyboy):
		return pyboy.game_wrapper().badges
