# CSC510 Foundations of Artificial Intelligence
# Informed Search Heuristics - Word Transformation Problem

# Import libraries
import heapq
from typing import List, Tuple, Set, Optional

# Define the Word Transformation class
class WordTransformationSearch:

# Initialize with a predefined dictionary of words
    def __init__(self):
        self.dictionary = {
            # -AT family (8 words)
            'cat', 'bat', 'rat', 'hat', 'mat', 'pat', 'sat', 'fat',
            # -AN family (6 words)  
            'can', 'ban', 'ran', 'fan', 'man', 'pan',
            # -ET family (8 words)
            'bet', 'get', 'jet', 'let', 'met', 'net', 'pet', 'set',
            # -OT family (6 words)
            'cot', 'dot', 'got', 'hot', 'lot', 'pot',
            # -IT family (5 words)
            'bit', 'fit', 'hit', 'kit', 'sit',
            # -UT family (4 words)
            'but', 'cut', 'hut', 'put',
            # -OG family (3 words)
            'cog', 'dog', 'log',
            # -UN family (3 words)
            'fun', 'gun', 'run',
            # -AD family (3 words)
            'bad', 'dad', 'mad',
            # -AG family (3 words)
            'bag', 'rag', 'tag',
            # Individual connectors (1 word)
            'dig'
        }
        
# Calculate 'hamming' distance between words    
    def hamming_distance(self, word1: str, word2: str) -> int:
        return sum(c1 != c2 for c1, c2 in zip(word1, word2))
# Gather all valid neighboring words - one letter different  
    def get_neighbors(self, word: str) -> List[str]:
        neighbors = []
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz': # all possible letters English alphabet
                if c != word[i]:
                    neighbor = word[:i] + c + word[i+1:]
                    if neighbor in self.dictionary:
                        neighbors.append(neighbor)
        return neighbors

# Define A* search implementation
# Returns: (path, nodes_explored) or None if no solution exists   
    def a_star_search(self, start: str, goal: str, verbose: bool = True) -> Optional[Tuple[List[str], int]]:
        if start not in self.dictionary or goal not in self.dictionary:
            print(f"Error: One or both words not in dictionary")
            return None
            
        if start == goal:
            return ([start], 0)
        
        # Priority queue: (f_score, g_score, current_word, path)
        frontier = [(self.hamming_distance(start, goal), 0, start, [start])]
        explored = set()
        nodes_explored = 0
        
        if verbose:
            print(f"\nA* Search: '{start}' to '{goal}'")
            print(f"Initial heuristic (Hamming distance): {self.hamming_distance(start, goal)}")
            print("\n--- Search Progress ---")
        
        while frontier:
            f_score, g_score, current, path = heapq.heappop(frontier)
            
            if current in explored:
                continue
                
            explored.add(current)
            nodes_explored += 1
            
            if verbose:
                print(f"Step {nodes_explored}: Exploring '{current}' (g={g_score}, h={f_score-g_score}, f={f_score})")
            
            if current == goal:
                if verbose:
                    print(f"\nSolution found! Path length: {len(path)-1} transformations")
                return (path, nodes_explored)
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor not in explored:
                    new_g_score = g_score + 1
                    h_score = self.hamming_distance(neighbor, goal)
                    new_f_score = new_g_score + h_score
                    new_path = path + [neighbor]
                    
                    heapq.heappush(frontier, (new_f_score, new_g_score, neighbor, new_path))
                    
                    if verbose and len(self.get_neighbors(current)) <= 5:  # Only show details for small branching
                        print(f"     Adding '{neighbor}' to frontier (g={new_g_score}, h={h_score}, f={new_f_score})")
        
        if verbose:
            print(f"\nNo solution found after exploring {nodes_explored} nodes")
        return None

# Display transformation path and details  
    def display_solution(self, path: List[str], goal: str):
        if not path:
            return
            
        print("Transformation Path ({} steps):".format(len(path)-1))
        print("=" * 40)
        
        for i, word in enumerate(path):
            if i == 0:
                print(f"Start: {word.upper()}")
            elif i == len(path) - 1:
                print(f"Goal:  {word.upper()}")
            else:
                # Show what changed
                prev_word = path[i-1]
                changed_pos = next(j for j in range(len(word)) if word[j] != prev_word[j])
                print(f"Step {i}: {word.upper()} (changed position {changed_pos}: {prev_word[changed_pos]} to {word[changed_pos]})")
        
        print(f"\nHeuristic check: Final Hamming distance = {self.hamming_distance(path[-1], goal)}")

# Define the main solve function    
    def solve(self, start: str, goal: str):
        print("=" * 50)
        print(f"Transforming '{start.upper()}' to '{goal.upper()}'")
        print("=" * 50)
        
        result = self.a_star_search(start, goal, verbose=True)
        
        if result:
            path, nodes_explored = result
            self.display_solution(path, goal)
            print(f"\nPerformance: {nodes_explored} nodes explored")
        else:
            print("No transformation path found!")


# Main execution
if __name__ == "__main__":
    solver = WordTransformationSearch()
    
    print("3 LETTER WORD TRANSFORMATION SEARCH")
    print("Transform one word into another!")
    print(f"Word dictionary: {', '.join(sorted(solver.dictionary))}")
    print(f"Total words: {len(solver.dictionary)}")
    print("="*50)
    print("Type 'quit' to exit")
    print()
    
# User interaction loop
    while True:
        try:
            start = input("\nEnter start word (3 letters): ").lower().strip()
            if start == 'quit':
                break
            if len(start) != 3:
                print("Please enter exactly 3 letters!")
                continue
            if start not in solver.dictionary:
                print(f"'{start}' not in word dictionary.")
                print(f"Try: {', '.join(sorted(list(solver.dictionary))[:8])}...")
                continue
                
            goal = input("Enter goal word (3 letters): ").lower().strip()
            if goal == 'quit':
                break
            if len(goal) != 3:
                print("Please enter exactly 3 letters!")
                continue
            if goal not in solver.dictionary:
                print(f"'{goal}' not in word dictionary.")
                print(f"Try: {', '.join(sorted(list(solver.dictionary))[:8])}...")
                continue
            
            solver.solve(start, goal)
            
# Handle unexpected errors
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThanks for using Word Transformation Search!")
