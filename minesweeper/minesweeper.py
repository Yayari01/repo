import itertools
import random
from copy import deepcopy
from collections import Counter


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        #TODO 1S
        Returns the set of all cells in self.cells known to be mines.
        """
        # if the mine count equals the number of cells then all of them are confirmed mines
        if self.count == len(self.cells):

            return self.cells
        else:
            return set()

    def known_safes(self):
        """
        #TODO 2S
        Returns the set of all cells in self.cells known to be safe.
        set.add(cell)
        """
        # if the mine count for a particular set equals 0 then all cells in that set are safe
        if self.count == 0:

            return self.cells

        else:

            return set()

    def mark_mine(self, cell):
        """
        #TODO 3S
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """

        # removing the cell that was passed in as an argument from the knowledge base and adjusts the count
        if cell in self.cells:

            self.cells.remove(cell)

            self.count -= 1

    def mark_safe(self, cell):
        """
        #TODO 4S
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        # removes the cell from the knowledge base based on the argument cell
        if cell in self.cells:

            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player

    .cell is the last clicked cell , and .cells would represent the neighbours
    then I can use .cells and count to represent the knowledge of neighbours and whether they are in contact
    with a mine in a format of {cells} = count
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        #TODO 1M
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """

        # Marking cell as move made
        self.moves_made.add(cell)

        # adding clicked on cell to safes and marking them as safes in the knowledgebase
        self.safes.add(cell)
        self.mark_safe(cell)

        # creating a set to collect the neighbours of the clicked on cell
        neighbours = set()

        # storing the coordinates in separate variables
        first_coordinate = cell[0]
        second_coordinate = cell[1]

        # a loop for finding neighbours around the clicked on cell
        # starting from -1 (behind the cell) ignoring the cell itself and then adding + 1 the one in front
        for i in range(-1, 2):

            for j in range(-1, 2):

                if i == 0 and j == 0:

                    continue

                # creating a tuple out of the surrounding cells
                new_tuple = (first_coordinate + i, second_coordinate + j)

                # if the tuple of neighbours is not out of bound then adding it to the neighbours set
                if 0 <= new_tuple[0] < self.height and 0 <= new_tuple[1] < self.width:

                    neighbours.add(new_tuple)

        # Checking whether there is a mine or a safe cell amongs the neighbours and adjusting the count based on that
        checked_neighbours = neighbours - self.safes - self.mines

        updated_count = count - len(neighbours & self.mines)

        # creating a sentence with the updated data
        object = Sentence(checked_neighbours, updated_count)

        # updating the knowledge with the new sentence as long as it's not empty
        if checked_neighbours:

            self.knowledge.append(object)

        # a while loop to iterate over each sentence objects , calling known safes and mines and then updating via
        # mark mine and mark safe
        while True:

            # storing the counts of the number of safe cells and mines
            safes_count = len(self.safes)
            mines_count = len(self.mines)

            # a copy of self.knowledge to use for loops to avoid errors due to potential size change
            # of the knowledge base during iterations
            knowledge_copy = self.knowledge.copy()

            # creating empty sets to collect known safes and mines to be used for inference
            known_safes = set()
            known_mines = set()

            # iterating over the copy of knowledge base
            for object in knowledge_copy:

                # calling the known_safes and known_mines methods to get the safe and mine cells
                known_safes.update(object.known_safes())
                known_mines.update(object.known_mines())

            # updating both the knowledge and the minesweeperAI gameplay data
            self.safes.update(known_safes)
            self.mines.update(known_mines)

            # Initialising and empty list to add sentences where all the cells have been removed
            to_remove = []

            # iterating over each safe cell and sentence from the knowledge base
            for safe_cell in known_safes:

                for sentence in self.knowledge:

                    # if the the cell from known safes is in the knowledge base then calling mark safe on that cell
                    if safe_cell in sentence.cells:

                        sentence.mark_safe(safe_cell)

                        # if there are no more cells then add the sentence to the to_remove list
                        if not sentence.cells and sentence not in to_remove:

                            to_remove.append(sentence)

            # loop iterating over each mine from known_mines along with each sentence from the knowledge base
            for mine_cell in known_mines:

                for sentence in self.knowledge:

                    # if the cell from known mines is found in the knowledge base then call mark mine on it
                    if mine_cell in sentence.cells:

                        sentence.mark_mine(mine_cell)

                        # if there are no more cells then add the sentence to the to_remove list
                        if not sentence.cells and sentence not in to_remove:

                            to_remove.append(sentence)

            # handling the removal of empty sentences gathered during the calling of mark_safe and mark_mine
            for sentence in to_remove:

                self.knowledge.remove(sentence)

            # another copy of self.knowledge just to be up to date with it's length and an empty list initialised for new sentences to be added
            knowledge_copy = self.knowledge.copy()
            new_sentences = []

            # inference nested loop
            for sentence1 in knowledge_copy:

                for sentence2 in knowledge_copy:

                    # condition for checking when one object is a subset of another and they are not the same object, also if the cells are the same
                    # which would mean now useful new knowledge would be create just an empty set which is not desirable, that's why the condition
                    # then it creates a new sentence object with the remaining cells and the adjusted count adding it to the knowledgebase
                    if sentence1 != sentence2 and sentence2.cells.issubset(sentence1.cells) and len(sentence1.cells) > len(sentence2.cells):

                        # adjusting the cell and mine count
                        remaining_cells = sentence1.cells - sentence2.cells
                        updated_count = sentence1.count - sentence2.count

                        # creating a new sentence with the adjusted cells and mine count
                        new_sentence = Sentence(remaining_cells, updated_count)

                        # condition checking if there are any remaining cells and whether the count is not negative and whether the new sentence
                        # is not already in self.knowledge
                        if remaining_cells and updated_count >= 0 and new_sentence not in self.knowledge and new_sentence not in new_sentences:

                            # collecting all the new sentences created in new_sentences
                            new_sentences.append(new_sentence)

            # condition checking the scenario if we have the complete knowledge of safe cells and mines and there were no new
            # sentences created during the inference , in that case we are breaking the loop here
            if len(self.safes) == safes_count and len(self.mines) == mines_count and len(new_sentences) == 0:

                break

            # updating the knowledge base with the new sentences
            self.knowledge.extend(new_sentences)

    def make_safe_move(self):
        """
        #TODO 2M
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """

        # iterating over each safe cell in self.safes
        for safe_move in self.safes:

            # if the cell is not amongs the moves made then return it as a move safe to be made
            if safe_move not in self.moves_made:

                return safe_move

        # if there is no single safe move that was not yet made then return None
        return None

    def make_random_move(self):
        """
        #TODO 3M
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        """
        if a safe move can't be made AI chooses at random
        """

        # a list collecting all the possible valid moves
        valid_moves = []

        # iterating over the board with a nested loop (for height and width of the board)
        for i in range(self.height):

            for j in range(self.width):

                move = (i, j)

                # checking each cell whether it's a possible move or not
                # if they are not in amongs the mines or moves already made
                if move not in self.mines and move not in self.moves_made:

                    valid_moves.append(move)

        # if the valid moves list is not empty then return a random move from valid moves using random
        if valid_moves:

            return random.choice(valid_moves)

        # if the list of valid moves is empty return None
        else:

            return None
