# Imports (standard library)
import json
import ssl
# Imports (3rd-party)
import websocket
import time 
# Imports (local application)
from constants import ACTION
from game_state import GameState
from game_state_wrapper import GameStateWrapper
from util import printf
# from hanabi_agent import LLMAgent
from llm_coordination_agents.hanabi_agent import LLMAgent, LLMAgentHanabiLive
import utils
import threading 

class HanabiClient:
    def __init__(self, url, cookie):
        # Initialize all class variables.
        self.commandHandlers = {}
        self.tables = {}
        self.username = ""
        self.ws = None
        self.url = url 
        self.games = {}
        self.initialized = False
        self.initial_draw_counter = 10
        game_config = {
            'agent_class': 'simple',
            'username': 'bot1',
            'num_human_players': 1,
            'num_total_players': 2,
            "players": ['player', 'bot1'],
            'empty_clues': False,
            'table_name': "AI_room",
            'table_pw': "",
            'variant': "No Variant",
            'num_episodes': 1,
            'life_tokens': 3,
            'info_tokens': 8,
            'deck_size': 50,
            'wait_move': 1,
            'colors': 5,
            'ranks': 5,
            'hand_size': 5,
            'max_moves': 2 * 5 + (2 - 1) * 5 + (2 - 1) * 5
        }
        self.stateHLE = GameStateWrapper(game_config)
        # Initialize the website command handlers (for the lobby).
        self.commandHandlers["welcome"] = self.welcome
        self.commandHandlers["warning"] = self.warning
        self.commandHandlers["error"] = self.error
        self.commandHandlers["chat"] = self.chat
        self.commandHandlers["table"] = self.table
        self.commandHandlers["tableList"] = self.table_list
        self.commandHandlers["tableGone"] = self.table_gone
        self.commandHandlers["tableStart"] = self.table_start

        # Initialize the website command handlers (for the game).
        self.commandHandlers["init"] = self.init
        # self.commandHandlers["gameHistory"] = self.game_history
        self.commandHandlers["gameAction"] = self.game_action
        self.commandHandlers["gameActionList"] = self.game_action_list
        self.commandHandlers["databaseID"] = self.database_id
        self.commandHandlers["yourTurn"] = self.your_turn

        # Start the WebSocket client.
        printf('Connecting to "' + url + '".')

        self.ws = websocket.WebSocketApp(
            url,
            on_message=lambda ws, message: self.websocket_message(ws, message),
            on_error=lambda ws, error: self.websocket_error(ws, error),
            on_open=lambda ws: self.websocket_open(ws),
            on_close=lambda ws: self.websocket_close(ws),
            cookie=cookie,
        )
        self.ws.run_forever(ping_interval=20, sslopt={"cert_reqs": ssl.CERT_NONE})

    # ------------------
    # WebSocket Handlers
    # ------------------

    def reconnect(self):
        try:
            # Wait for a short period to avoid aggressive reconnection attempts
            time.sleep(5)  # Wait for 5 seconds before trying to reconnect
            # Reinitialize the WebSocket connection
            self.ws = websocket.WebSocketApp(
                self.url,
                on_message=lambda ws, message: self.websocket_message(ws, message),
                on_error=lambda ws, error: self.websocket_error(ws, error),
                on_open=lambda ws: self.websocket_open(ws),
                on_close=lambda ws: self.websocket_close(ws),
                cookie=self.cookie,
            )
            self.ws.run_forever()
        except Exception as e:
            printf(f"Reconnection failed: {e}")

    def websocket_message(self, ws, message):
        # WebSocket messages from the server come in the format of:
        # commandName {"fieldName":"value"}
        # For more information, see:
        # https://github.com/Zamiell/hanabi-live/blob/master/src/websocketMessage.go
        result = message.split(" ", 1)  # Split it into two things
        if len(result) != 1 and len(result) != 2:
            printf("error: received an invalid WebSocket message:")
            printf(message)
            return

        command = result[0]
        try:
            data = json.loads(result[1])
        except:
            printf(
                'error: the JSON data for the command of "' + command + '" was invalid'
            )
            return

        if command in self.commandHandlers:
            printf('debug: got command "' + command + '"')
            try:
                self.commandHandlers[command](data)
            except Exception as e:
                printf('error: command handler for "' + command + '" failed:', e)
                return
        else:
            printf('debug: ignoring command "' + command + '"')

    def websocket_error(self, ws, error):
        printf("Encountered a WebSocket error:", error)
        self.reconnect()

    def websocket_close(self, ws):
        printf("WebSocket connection closed.")

    def websocket_open(self, ws):
        printf("Successfully established WebSocket connection.")

    # --------------------------------
    # Website Command Handlers (Lobby)
    # --------------------------------

    def welcome(self, data):
        # The "welcome" message is the first message that the server sends us
        # once we have established a connection. It contains our username,
        # settings, and so forth.
        self.username = data["username"]
        # if "playingAtTables" in data:
        #     for table_id in data["playingAtTables"]:
        #         print("rejoining table: ", table_id)
        #         self.send(
        #             "tableReattend",
        #             {
        #                 "tableID": table_id,
        #             },
        #         )

    def error(self, data):
        # Either we have done something wrong, or something has gone wrong on
        # the server.
        printf(data)

    def warning(self, data):
        # We have done something wrong.
        printf(data)

    def chat(self, data):
        # We only care about private messages.
        if data["recipient"] != self.username:
            return

        # We only care about private messages that start with a forward slash.
        if not data["msg"].startswith("/"):
            return
        data["msg"] = data["msg"][1:]  # Remove the slash.

        # We want to split it into two things.
        result = data["msg"].split(" ", 1)
        command = result[0]

        if command == "join":
            self.chat_join(data)
        else:
            msg = "That is not a valid command."
            self.chat_reply(msg, data["who"])

    def chat_join(self, data):
        # Someone sent a private message to the bot and requested that we join
        # their game. Find the table that the current user is currently in.
        table_id = None
        for table in self.tables.values():
            # Ignore games that have already started (and shared replays).
            if table["running"]:
                continue

            if data["who"] in table["players"]:
                if len(table["players"]) == 6:
                    msg = "Your game is full. Please make room for me before requesting that I join your game."
                    self.chat_reply(msg, data["who"])
                    return

                table_id = table["id"]
                break

        if table_id is None:
            msg = "Please create a table first before requesting that I join your game."
            self.chat_reply(msg, data["who"])
            return

        self.send(
            "tableJoin",
            {
                "tableID": table_id,
            },
        )

    def table(self, data):
        
        self.tables[data["id"]] = data

    def table_list(self, data_list):
        for data in data_list:
            self.table(data)

    def table_gone(self, data):
        del self.tables[data["tableID"]]

    def table_start(self, data):
        # The server has told us that a game that we are in is starting. So,
        # the next step is to request some high-level information about the
        # game (e.g. number of players). The server will respond with an "init"
        # command.
        self.send(
            "getGameInfo1",
            {
                "tableID": data["tableID"],
            },
        )

    # -------------------------------
    # Website Command Handlers (Game)
    # -------------------------------

    def game_history(self, data):
        # The server has sent us a list of all of the actions that have taken
        # place thus far in the game.
        # for action in data["list"]:
        #     self.handle_action(action, data["tableID"])
        state = GameState()
        self.stateHLE.init_players_from_game_history(data) 
        self.games[data["tableID"]] = state
        state.player_names = data["playerNames"]
        state.our_player_index = data["ourPlayerIndex"]
        self.agent = LLMAgentHanabiLive(state.our_player_index)



        # Initialize the hands for each player (an array of cards).
        for _ in range(len(state.player_names)):
            state.hands.append([])

        # Initialize the play stacks.
        """
        This is hard coded to 5 because there 5 suits in a no variant game
        The website supports variants that have 3, 4, and 6 suits
        TODO This code should compare "data['variant']" to the "variants.json"
        file in order to determine the correct amount of suits
        https://raw.githubusercontent.com/Zamiell/hanabi-live/master/public/js/src/data/variants.json
        """
        num_suits = 5
        for _ in range(num_suits):
            state.play_stacks.append([])

        # At this point, the JavaScript client would have enough information to
        # load and display the game UI. For our purposes, we do not need to
        # load a UI, so we can just jump directly to the next step. Now, we
        # request the specific actions that have taken place thus far in the
        # game (which will come in a "gameActionList").
        self.send(
            "getGameInfo2",
            {
                "tableID": data["tableID"],
            },
        )
        
        self.send(
            "getGameInfo2",
            {
                "tableID": list(self.tables.keys())[0],
            },
        )
    def init(self, data):
        # At the beginning of the game, the server sends us some high-level
        # data about the game, including the names and ordering of the players
        # at the table.

        # Make a new game state and store it on the "games" dictionary.
        self.initialized = True 
        state = GameState()
        self.stateHLE.init_players(data) 
        self.games[data["tableID"]] = state
        state.player_names = data["names"]
        state.our_player_index = data["seat"]
        self.agent = LLMAgentHanabiLive(state.our_player_index)



        # Initialize the hands for each player (an array of cards).
        for _ in range(len(state.player_names)):
            state.hands.append([])

        # Initialize the play stacks.
        """
        This is hard coded to 5 because there 5 suits in a no variant game
        The website supports variants that have 3, 4, and 6 suits
        TODO This code should compare "data['variant']" to the "variants.json"
        file in order to determine the correct amount of suits
        https://raw.githubusercontent.com/Zamiell/hanabi-live/master/public/js/src/data/variants.json
        """
        num_suits = 5
        for _ in range(num_suits):
            state.play_stacks.append([])

        # At this point, the JavaScript client would have enough information to
        # load and display the game UI. For our purposes, we do not need to
        # load a UI, so we can just jump directly to the next step. Now, we
        # request the specific actions that have taken place thus far in the
        # game (which will come in a "gameActionList").
        self.send(
            "getGameInfo2",
            {
                "tableID": data["tableID"],
            },
        )

    def game_action(self, data):
        state = self.games[data["tableID"]]

        # We just received a new action for an ongoing game.
        self.handle_action(data["action"], data["tableID"])
        # print("CURRENT PLAYER: ", state.current_player_index)
        # print("OUR PLAYER: ", state.our_player_index)
        # if state.current_player_index == state.our_player_index:
        #     # self.decide_action(data["tableID"])
        #     thread = threading.Thread(target=self.decide_action, args=(data['tableID'],))
        #     thread.start()

    def game_action_list(self, data):
        state = self.games[data["tableID"]]

        # We just received a list of all of the actions that have occurred thus
        # far in the game.
        for action in data["list"]:
            self.handle_action(action, data["tableID"])
        # print("CURRENT PLAYER: ", state.current_player_index)
        # print("OUR PLAYER: ", state.our_player_index)
        # if state.current_player_index == state.our_player_index:
        #     thread = threading.Thread(target=self.decide_action, args=(data['tableID'],))
        #     thread.start()
            
            # self.decide_action(data["tableID"])

        # Let the server know that we have finished "loading the UI" (so that
        # our name does not appear as red / disconnected).
        self.send(
            "loaded",
            {
                "tableID": data["tableID"],
            },
        )


        # Let the server know that we have finished "loading the UI" (so that
        # our name does not appear as red / disconnected).
        self.send(
            "loaded",
            {
                "tableID": data["tableID"],
            },
        )

    def handle_action(self, data, table_id):
        printf(
            'debug: got a game action of "%s" for table %d' % (data["type"], table_id)
        )
        print("CURRENT DATA IS:    ", data)
        state = self.games[table_id]
        self.stateHLE.update_state(data)
        # self.stateHLE.get_agent_observation()
        if data["type"] == "draw":
            # Add the newly drawn card to the player's hand.
            hand = state.hands[data["who"]]
            hand.append(
                {
                    "order": data["order"],
                    "suit_index": data["suit"],
                    "rank": data["rank"],
                }
            )
            # state.our_player_index = data["playerIndex"]
            # The first player to draw cards will be the first player to play 
            # if self.initial_draw_counter > -1:
            #     self.initial_draw_counter -= 1
            # if self.initial_draw_counter == 0:

            # if self.stateHLE.initial_draw_counter == 0:
            #     # Finished drawing all cards, the player who drew first must have first turn 
            #     state.current_player_index = 1 - data['playerIndex']
            #     self.stateHLE.initial_draw_counter = -1

        elif data["type"] == "play":
            player_index = data['which']["index"]
            order = data['which']["order"]
            card = self.remove_card_from_hand(state, player_index, order)
            if card is not None:
                # TODO Add the card to the play stacks.
                pass

        elif data["type"] == "discard":
            player_index = data['which']["index"]
            order = data['which']["order"]
            card = self.remove_card_from_hand(state, player_index, order)
            if card is not None:
                # TODO Add the card to the discard stacks.
                pass

            # Discarding adds a clue. But misplays are represented as discards,
            # and misplays do not grant a clue.
            if not data["failed"]:
                state.clue_tokens += 1

        elif data["type"] == "clue":
            # Each clue costs one clue token.
            state.clue_tokens -= 1

            # TODO We might also want to update the state of cards that are
            # "touched" by the clue so that we can keep track of the positive
            # and negative information "on" the card.
            

        elif data["type"] == "turn":
            # A turn is comprised of one or more game actions (e.g. play +
            # draw). The turn action will be the final thing sent on a turn,
            # which also includes the index of the new current player.
            # TODO: This action may be removed from the server in the future
            # since the client is expected to calculate the turn on its own
            # from the actions.
            state.turn = data["num"]
            state.current_player_index = data["who"]

    def database_id(self, data):
        # Games are transformed into shared replays after they are completed.
        # The server sends a "databaseID" message when the game has ended. Use
        # this as a signal to leave the shared replay.
        self.send(
            "tableUnattend",
            {
                "tableID": data["tableID"],
            },
        )

        # Delete the game state for the game to free up memory.
        del self.games[data["tableID"]]

    # ------------
    # AI functions
    # ------------

    def your_turn(self, data):
        action_thread = threading.Thread(target=self.decide_action, args=(data['tableID'],))
        action_thread.start()
    
    def decide_action(self, table_id):
        try: 
            state = self.games[table_id]
            stateHLE = self.stateHLE
            observation = stateHLE.get_agent_observation()
            print("OBSERVATION IS: ", observation)
            partner_move = None
            previous_move_hanabi = observation['last_moves'][0].move()
            state.current_player_index = -1
            if 'DEAL' in str(previous_move_hanabi._type):
                previous_move_hanabi = observation['last_moves'][1].move()

            if 'REVEAL_COLOR' in str(previous_move_hanabi._type):
                partner_move = {'action_type': 'REVEAL_COLOR', 'target_offset': previous_move_hanabi.target_offset(), 'color': previous_move_hanabi.color()}
            elif 'REVEAL_RANK' in str(previous_move_hanabi._type):
                partner_move = {'action_type': 'REVEAL_RANK', 'target_offset': previous_move_hanabi.target_offset(), 'rank': previous_move_hanabi.rank()}
            elif 'PLAY' in str(previous_move_hanabi._type):
                partner_move = {'action_type': 'PLAY', 'color': previous_move_hanabi.color(), 'rank': previous_move_hanabi.rank()}
            elif 'DISCARD' in str(previous_move_hanabi._type):
                partner_move = {'action_type': 'DISCARD', 'color': previous_move_hanabi.color(), 'rank': previous_move_hanabi.rank()}
            
            selected_move = self.agent.get_next_move(observation, partner_move)
            # selected_move = observation['legal_moves'][-3]
            print("SELECTED MOVE IS: ", selected_move)
            # {"tableID": 2, "type": 2, "target": 1, "value": "B"} 
            if selected_move['action_type'] == 'PLAY':
                print('SELECTED ACTION IS: PLAY', ACTION.PLAY)
                self.send(
                    "action",
                    {
                        "tableID": table_id,
                        "type": ACTION.PLAY,
                        "target": state.hands[state.our_player_index][selected_move['card_index']]['order'],
                    },
                )
                
            elif selected_move['action_type'] == 'DISCARD':
                print('SELECTED ACTION IS: DISCARD', ACTION.DISCARD)
                self.send(
                    "action",
                    {
                        "tableID": table_id,
                        "type": ACTION.DISCARD,
                        "target": state.hands[state.our_player_index][selected_move['card_index']]['order'],
                    },
                )
                
            elif selected_move['action_type'] == 'REVEAL_COLOR':
                target_idx = (state.our_player_index + selected_move["target_offset"]) % 2
                print('SELECTED ACTION IS: COLOR CLUE', ACTION.COLOR_CLUE)
                self.send(
                    "action",
                    {
                        "tableID": table_id,
                        "type": ACTION.COLOR_CLUE,
                        "target": target_idx,
                        "value": utils.convert_color(selected_move['color']),
                    },
                )
                
            else:
                target_idx = (state.our_player_index + selected_move["target_offset"]) % 2
                print('SELECTED ACTION IS: RANK CLUE', ACTION.RANK_CLUE)
                self.send(
                    "action",
                    {
                        "tableID": table_id,
                        "type": ACTION.RANK_CLUE,
                        "target": target_idx,
                        "value": selected_move['rank'] + 1,
                    },
                )
                
        except Exception as e:
            printf(f"Error in perform_api_call: {e}")

        
    # -----------
    # Subroutines
    # -----------

    def chat_reply(self, message, recipient):
        self.send(
            "chatPM",
            {
                "msg": message,
                "recipient": recipient,
                "room": "lobby",
            },
        )

    def send(self, command, data):
        if not isinstance(data, dict):
            data = {}
        self.ws.send(command + " " + json.dumps(data))
        printf('debug: sent command "' + command + '"')

    def remove_card_from_hand(self, state, player_index, order):
        hand = state.hands[player_index]
        card_index = -1
        for i in range(len(hand)):
            card = hand[i]
            if card["order"] == order:
                card_index = i
        if card_index == -1:
            printf(
                "error: unable to find card with order " + str(order) + " in"
                "the hand of player " + str(player_index)
            )
            return None
        card = hand[card_index]
        del hand[card_index]
        return card