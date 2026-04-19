# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


# ──────────────────────────────────────────────────────────────
#  Question 1 — Minimax
# ──────────────────────────────────────────────────────────────

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent.
            agentIndex=0 means Pacman, ghosts are >= 1.

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action.

          gameState.getNumAgents():
            Returns the total number of agents in the game.
        """
        action, score = self.minimax(0, 0, gameState)
        return action

    def minimax(self, curr_depth, agent_index, gameState):
        '''
        Returns the best action and score for the current agent using minimax.
        - MAX player (agent_index == 0, Pacman)  : maximises score.
        - MIN player (agent_index >= 1, Ghosts)  : minimises score.
        Recursion ends when max depth is reached or no legal actions exist.

        :param curr_depth:   current depth in the search tree (int)
        :param agent_index:  index of the agent currently acting (int)
        :param gameState:    current game state (GameState)
        :return:             (best_action, best_score)
        '''
        # Roll over agent index and increase depth once every agent has moved
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1

        # Terminal condition: reached max depth — evaluate the leaf state
        if curr_depth == self.depth:
            return None, self.evaluationFunction(gameState)

        best_score, best_action = None, None

        if agent_index == 0:  # MAX player — Pacman
            for action in gameState.getLegalActions(agent_index):
                next_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.minimax(curr_depth, agent_index + 1, next_state)
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action

        else:  # MIN player — Ghost
            for action in gameState.getLegalActions(agent_index):
                next_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.minimax(curr_depth, agent_index + 1, next_state)
                if best_score is None or score < best_score:
                    best_score = score
                    best_action = action

        # Leaf state with no legal actions — evaluate directly
        if best_score is None:
            return None, self.evaluationFunction(gameState)

        return best_action, best_score


# ──────────────────────────────────────────────────────────────
#  Question 2 — Alpha-Beta Pruning
# ──────────────────────────────────────────────────────────────

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction,
          computed efficiently with alpha-beta pruning.
        """
        # Initialise alpha = -inf (worst possible for MAX)
        #            beta  = +inf (worst possible for MIN)
        action, score = self.alphabeta(0, 0, gameState, float('-inf'), float('inf'))
        return action

    def alphabeta(self, curr_depth, agent_index, gameState, alpha, beta):
        '''
        Returns the best action and score using minimax with alpha-beta pruning.

        Alpha = highest score MAX can already guarantee on the current path.
        Beta  = lowest  score MIN can already guarantee on the current path.

        Pruning uses STRICT inequality (alpha > beta).
        Do NOT prune on equality (alpha >= beta) — required by the autograder.

        :param curr_depth:   current depth in the search tree (int)
        :param agent_index:  index of the agent currently acting (int)
        :param gameState:    current game state (GameState)
        :param alpha:        best score MAX can guarantee so far (float)
        :param beta:         best score MIN can guarantee so far (float)
        :return:             (best_action, best_score)
        '''
        # Roll over agent index and increase depth once every agent has moved
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1

        # Terminal condition: reached max depth — evaluate the leaf state
        if curr_depth == self.depth:
            return None, self.evaluationFunction(gameState)

        best_score, best_action = None, None

        if agent_index == 0:  # MAX player — Pacman
            for action in gameState.getLegalActions(agent_index):
                next_state = gameState.generateSuccessor(agent_index, action)

                # Recurse — next agent is the first ghost (agent_index + 1)
                _, score = self.alphabeta(curr_depth, agent_index + 1, next_state, alpha, beta)

                # Keep track of the best (highest) score for MAX
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action

                # Update alpha: the best MAX can guarantee on this path
                alpha = max(alpha, best_score)

                # Beta cut-off — MIN would never let us reach a node this good,
                # so stop exploring remaining siblings.
                # IMPORTANT: strict inequality only — do NOT use >=
                if alpha > beta:
                    break

        else:  # MIN player — Ghost
            for action in gameState.getLegalActions(agent_index):
                next_state = gameState.generateSuccessor(agent_index, action)

                # Recurse — next agent is the next ghost or Pacman (agent_index + 1)
                _, score = self.alphabeta(curr_depth, agent_index + 1, next_state, alpha, beta)

                # Keep track of the best (lowest) score for MIN
                if best_score is None or score < best_score:
                    best_score = score
                    best_action = action

                # Update beta: the best MIN can guarantee on this path
                beta = min(beta, best_score)

                # Alpha cut-off — MAX would never let us reach a node this bad,
                # so stop exploring remaining siblings.
                # IMPORTANT: strict inequality only — do NOT use >=
                if alpha > beta:
                    break

        # Leaf state with no legal actions — evaluate directly
        if best_score is None:
            return None, self.evaluationFunction(gameState)

        return best_action, best_score


# ──────────────────────────────────────────────────────────────
#  Question 4 — Expectimax  (not yet implemented)
# ──────────────────────────────────────────────────────────────

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction.
          All ghosts should be modelled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


# ──────────────────────────────────────────────────────────────
#  Question 5 — Better Evaluation Function  (not yet implemented)
# ──────────────────────────────────────────────────────────────

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
