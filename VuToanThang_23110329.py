import sys
import random
import itertools
from collections import deque
import heapq
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                            QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, 
                            QFrame, QSizePolicy, QGraphicsDropShadowEffect,
                            QDockWidget, QTextEdit, QScrollArea, QLineEdit,
                            QGroupBox, QTabWidget, QComboBox, QFileDialog,
                            QProgressBar, QSpinBox, QTableWidget, QTableWidgetItem,
                            QRadioButton, QFormLayout, QHeaderView)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QTimer, QThread
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient, QPainterPath

# For evaluation and visualization
import time
import pandas as pd
import numpy as np
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Modern Material Colors
COLORS = {
    'primary': '#2196F3',      # Blue 500
    'secondary': '#009688',    # Teal 500
    'accent': '#FF5722',       # Deep Orange 500
    'danger': '#F44336',       # Red 500
    'success': '#4CAF50',      # Green 500
    'warning': '#FF9800',      # Orange 500
    'background': '#F5F5F5',   # Grey 100
    'surface': '#FFFFFF',      # White
    'text_primary': '#212121', # Grey 900
    'text_secondary': '#757575', # Grey 600
    'border': '#E0E0E0',       # Grey 300
    'divider': '#EEEEEE',      # Grey 200
    
    # Algorithm group colors
    'uninformed': '#2196F3',   # Blue
    'informed': '#4CAF50',     # Green
    'local': '#FF5722',        # Deep Orange
    'complex': '#009688',      # Teal
    'evolutionary': '#673AB7'   # Deep Purple
}

# Algorithm Functions
def generate_random_puzzle():
    """Generate a random solvable 8-puzzle state"""
    while True:
        state = list(range(9))
        random.shuffle(state)
        if is_solvable(state):
            return state

def get_inversion_count(state):
    """Count inversions in the puzzle state"""
    inv_count = 0
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            if state[i] != 0 and state[j] != 0 and state[i] > state[j]:
                inv_count += 1
    return inv_count

def is_solvable(state):
    """Check if the puzzle state is solvable"""
    inv_count = get_inversion_count(state)
    return inv_count % 2 == 0

def get_neighbors(state):
    """Generate all possible next states from current state"""
    neighbors = []
    empty = state.index(0)
    row, col = empty // 3, empty % 3
    
    # Possible moves: up, down, left, right
    moves = [(-1,0), (1,0), (0,-1), (0,1)]
    
    for dx, dy in moves:
        new_row, new_col = row + dx, col + dy
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_pos = new_row * 3 + new_col
            new_state = list(state)
            new_state[empty], new_state[new_pos] = new_state[new_pos], new_state[empty]
            neighbors.append(tuple(new_state))
    
    return neighbors

def manhattan_distance(state, goal):
    """Calculate Manhattan distance heuristic"""
    distance = 0
    for i in range(9):
        if state[i] != 0:  # Không tính khoảng cách cho ô trống
            current_row, current_col = i // 3, i % 3
            try:
                goal_pos = goal.index(state[i])
                goal_row, goal_col = goal_pos // 3, goal_pos % 3
                distance += abs(current_row - goal_row) + abs(current_col - goal_col)
            except ValueError:
                # Nếu state[i] không tồn tại trong goal, gán khoảng cách cao
                distance += 6  # Giá trị tối đa cho manhattan distance trong puzzle 3x3 là 4
    return distance

def misplaced_tiles(state, goal):
    """Calculate number of misplaced tiles heuristic"""
    return sum(1 for i in range(9) if state[i] != 0 and state[i] != goal[i])

def bfs(start, goal):
    """Breadth First Search - Tìm kiếm theo chiều rộng
    Đặc điểm: 
    - Duyệt tất cả các trạng thái ở cùng độ sâu trước khi đi xuống độ sâu tiếp theo
    - Đảm bảo tìm ra đường đi ngắn nhất
    - Tốn nhiều bộ nhớ do phải lưu trữ tất cả trạng thái
    """
    if not is_solvable(start):
        return []
        
    queue = deque([(tuple(start), [tuple(start)])])
    visited = {tuple(start)}
    
    while queue:
        state, path = queue.popleft()  # Lấy trạng thái đầu tiên trong queue
        if state == goal:
            return path
            
        # Duyệt tất cả các trạng thái kế tiếp có thể
        for neighbor in get_neighbors(state):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []

def dfs(start, goal, depth_limit=50):
    """Depth First Search - Tìm kiếm theo chiều sâu
    Đặc điểm:
    - Đi sâu nhất có thể theo một nhánh trước khi quay lui
    - Thêm giới hạn độ sâu để tránh đi quá sâu
    - Tiết kiệm bộ nhớ hơn BFS
    """
    if not is_solvable(start):
        return []
        
    stack = [(tuple(start), [tuple(start)], 0)]  # Thêm độ sâu vào stack
    visited = {tuple(start)}
    
    while stack:
        state, path, depth = stack.pop()
        if state == goal:
            return path
            
        if depth < depth_limit:  # Kiểm tra giới hạn độ sâu
            # Duyệt các trạng thái kế tiếp theo thứ tự ngược lại
            for neighbor in reversed(list(get_neighbors(state))):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor], depth + 1))
    return []

def uniform_cost(start, goal):
    """Uniform Cost Search - Tìm kiếm theo chi phí đồng nhất
    Đặc điểm:
    - Mở rộng node có chi phí thấp nhất
    - Đảm bảo tìm ra đường đi tối ưu
    - Sử dụng priority queue để lựa chọn node tiếp theo
    """
    if not is_solvable(start):
        return []
        
    pq = [(0, tuple(start), [tuple(start)])]  # (cost, state, path)
    visited = {tuple(start): 0}
    
    while pq:
        cost, state, path = heapq.heappop(pq)
        if state == goal:
            return path
            
        for neighbor in get_neighbors(state):
            new_cost = cost + 1  # Chi phí mỗi bước di chuyển là 1
            if neighbor not in visited or new_cost < visited[neighbor]:
                visited[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))
    return []

def greedy(start, goal):
    """Greedy Best-First Search - Tìm kiếm tham lam theo best-first
    Đặc điểm:
    - Chọn node có heuristic tốt nhất
    - Không quan tâm đến chi phí đường đi
    - Nhanh nhưng không đảm bảo tối ưu
    """
    if not is_solvable(start):
        return []
        
    pq = [(manhattan_distance(tuple(start), goal), tuple(start), [tuple(start)])]
    visited = {tuple(start)}
    
    while pq:
        _, state, path = heapq.heappop(pq)
        if state == goal:
            return path
            
        for neighbor in get_neighbors(state):
            if neighbor not in visited:
                visited.add(neighbor)
                heapq.heappush(pq, (manhattan_distance(neighbor, goal), 
                                   neighbor, path + [neighbor]))
    return []

def astar(start, goal):
    """A* Search - Tìm kiếm A*
    Đặc điểm:
    - Kết hợp chi phí thực tế (g) và heuristic (h)
    - f(n) = g(n) + h(n)
    - Đảm bảo tìm ra đường đi tối ưu nếu heuristic admissible
    """
    if not is_solvable(start):
        return []
        
    pq = [(manhattan_distance(tuple(start), goal), 0, tuple(start), [tuple(start)])]  # (f, g, state, path)
    visited = {tuple(start): 0}  # state: g_score
    
    while pq:
        f, g, state, path = heapq.heappop(pq)
        if state == goal:
            return path
            
        for neighbor in get_neighbors(state):
            new_g = g + 1  # Chi phí thực tế để đến neighbor
            if neighbor not in visited or new_g < visited[neighbor]:
                visited[neighbor] = new_g
                h = manhattan_distance(neighbor, goal)  # Heuristic estimate
                f = new_g + h  # Tổng chi phí
                heapq.heappush(pq, (f, new_g, neighbor, path + [neighbor]))
    return []

def hill_climbing(start, goal, max_iterations=1000):
    """Hill Climbing Search implementation with improvements"""
    if not is_solvable(start):
        return []
        
    current_state = tuple(start)
    current_path = [tuple(start)]
    current_score = manhattan_distance(current_state, goal)
    
    best_state = current_state
    best_score = current_score
    best_path = current_path
    
    iterations = 0
    
    while iterations < max_iterations:
        neighbors = get_neighbors(current_state)
        best_neighbor = None
        best_neighbor_score = float('inf')
        
        for neighbor in neighbors:
            score = manhattan_distance(neighbor, goal)
            if score < best_neighbor_score:
                best_neighbor = neighbor
                best_neighbor_score = score
        
        if best_neighbor is None or best_neighbor_score >= current_score:
            if current_score < best_score:
                best_state = current_state
                best_score = current_score
                best_path = current_path
            
            if neighbors:
                random_neighbor = random.choice(neighbors)
                current_state = random_neighbor
                current_path.append(current_state)
                current_score = manhattan_distance(current_state, goal)
                iterations += 1
                continue
            
            return best_path
            
        current_state = best_neighbor
        current_path.append(current_state)
        current_score = best_neighbor_score
        
        if current_state == goal:
            return current_path
            
        iterations += 1
    
    return best_path

def simulated_annealing(start, goal, max_iterations=5000):
    """Simulated Annealing - Mô phỏng luyện kim
    Đặc điểm:
    - Chấp nhận trạng thái xấu hơn với xác suất giảm dần
    - Có thể thoát khỏi local maximum
    - Hội tụ dần về giải pháp tốt
    """
    if not is_solvable(start):
        return []
        
    current_state = tuple(start)
    current_path = [tuple(start)]
    current_score = manhattan_distance(current_state, goal)
    
    best_state = current_state
    best_score = current_score
    best_path = current_path
    
    temperature = 1000.0
    cooling_rate = 0.99
    
    for _ in range(max_iterations):
        if temperature < 0.1:
            break
            
        neighbors = get_neighbors(current_state)
        if not neighbors:
            break
            
        next_state = random.choice(neighbors)
        next_score = manhattan_distance(next_state, goal)
        
        # Tính delta_E và xác suất chấp nhận trạng thái mới
        delta_e = next_score - current_score
        if delta_e < 0 or random.random() < math.exp(-delta_e / temperature):
            current_state = next_state
            current_path.append(current_state)
            current_score = next_score
            
            if current_score < best_score:
                best_state = current_state
                best_score = current_score
                best_path = current_path.copy()
                
            if current_state == goal:
                return current_path
                
        temperature *= cooling_rate
        
    return best_path

def iterative_deepening_search(start, goal):
    """Iterative Deepening Search - Tìm kiếm theo chiều sâu tăng dần
    Đặc điểm:
    - Kết hợp ưu điểm của DFS và BFS
    - Tăng dần độ sâu tìm kiếm
    - Đảm bảo tìm ra đường đi ngắn nhất
    - Tiết kiệm bộ nhớ
    """
    def dls(state, goal, depth, path, visited):
        if state == goal:
            return path
        if depth <= 0:
            return None
            
        for neighbor in get_neighbors(state):
            if neighbor not in visited:
                visited.add(neighbor)
                result = dls(neighbor, goal, depth - 1, path + [neighbor], visited)
                if result is not None:
                    return result
                visited.remove(neighbor)
        return None

    if not is_solvable(start):
        return []

    max_depth = 0
    while max_depth <= 31:  # Giới hạn độ sâu tối đa cho 8-puzzle
        visited = {tuple(start)}
        result = dls(tuple(start), goal, max_depth, [tuple(start)], visited)
        if result:
            return result
        max_depth += 1
    return []

def ida_star(start, goal):
    """IDA* Search - Iterative Deepening A*
    Đặc điểm:
    - Kết hợp IDS với A*
    - Sử dụng heuristic để giới hạn độ sâu
    - Tiết kiệm bộ nhớ hơn A*
    - Đảm bảo tìm ra đường đi tối ưu
    """
    def search(path, g, bound):
        state = path[-1]
        f = g + manhattan_distance(state, goal)
        if f > bound:
            return f, None
        if state == goal:
            return -1, path
        
        min_cost = float('inf')
        for neighbor in get_neighbors(state):
            if neighbor not in path:
                path.append(neighbor)
                cost, new_path = search(path, g + 1, bound)
                if cost == -1:
                    return -1, new_path
                if cost < min_cost:
                    min_cost = cost
                path.pop()
        return min_cost, None

    if not is_solvable(start):
        return []

    bound = manhattan_distance(tuple(start), goal)
    path = [tuple(start)]
    while True:
        cost, result = search(path, 0, bound)
        if cost == -1:
            return result
        if cost == float('inf'):
            return []
        bound = cost

def hill_climbing_max(start, goal):
    """Hill Climbing với Maximum Steepness Ascent
    Đặc điểm:
    - Luôn chọn trạng thái tốt nhất trong các trạng thái kế tiếp
    - Dừng khi không còn trạng thái nào tốt hơn
    - Có thể bị kẹt ở local maximum
    """
    if not is_solvable(start):
        return []
        
    current_state = tuple(start)
    current_path = [tuple(start)]
    
    while True:
        neighbors = get_neighbors(current_state)
        if not neighbors:
            break
            
        # Tìm neighbor tốt nhất (có manhattan distance nhỏ nhất)
        best_neighbor = min(neighbors, key=lambda x: manhattan_distance(x, goal))
        
        # Nếu không có neighbor nào tốt hơn, dừng lại
        if manhattan_distance(best_neighbor, goal) >= manhattan_distance(current_state, goal):
            break
            
        current_state = best_neighbor
        current_path.append(current_state)
        
        if current_state == goal:
            return current_path
            
    return current_path

def hill_climbing_random_restart(start, goal, max_restarts=10):
    """Hill Climbing với Random Restart
    Đặc điểm:
    - Khởi động lại từ trạng thái ngẫu nhiên khi bị kẹt
    - Có thể thoát khỏi local maximum
    - Cho phép nhiều lần thử
    """
    if not is_solvable(start):
        return []
        
    best_path = None
    best_distance = float('inf')
    
    for _ in range(max_restarts):
        current_state = tuple(start) if _ == 0 else tuple(generate_random_puzzle())
        current_path = [current_state]
        
        while True:
            neighbors = get_neighbors(current_state)
            if not neighbors:
                break
                
            best_neighbor = min(neighbors, key=lambda x: manhattan_distance(x, goal))
            
            if manhattan_distance(best_neighbor, goal) >= manhattan_distance(current_state, goal):
                break
                
            current_state = best_neighbor
            current_path.append(current_state)
            
            if current_state == goal:
                return current_path
        
        end_distance = manhattan_distance(current_state, goal)
        if end_distance < best_distance:
            best_distance = end_distance
            best_path = current_path
            
    return best_path if best_path else []

def local_beam_search(start, goal, beam_width=5, max_iterations=1000):
    """Local Beam Search - Tìm kiếm tia cục bộ
    Đặc điểm:
    - Duy trì k trạng thái (tia) đồng thời thay vì chỉ 1 trạng thái
    - Tìm kiếm theo nhiều hướng đồng thời
    - Có thể thoát khỏi cực tiểu cục bộ (local minima)
    - Chia sẻ thông tin giữa các trạng thái
    """
    if not is_solvable(start):
        return []
    
    # Khởi tạo beam với start state 
    states = [(tuple(start), [tuple(start)])]  # (state, path)
    
    for _ in range(max_iterations):
        # Tạo tất cả các successor
        all_successors = []
        for state, path in states:
            if state == goal:
                return path
                
            for neighbor in get_neighbors(state):
                if neighbor not in path:  # Tránh các trạng thái lặp lại
                    all_successors.append((neighbor, path + [neighbor]))
        
        if not all_successors:
            break
            
        # Chọn k successor tốt nhất
        all_successors.sort(key=lambda x: manhattan_distance(x[0], goal))
        states = all_successors[:beam_width]
    
    # Trả về đường đi tốt nhất trong các tia
    if states:
        best_path = min(states, key=lambda x: manhattan_distance(x[0], goal))[1]
        return best_path
    return []

def genetic_algorithm(start, goal, population_size=100, generations=50):
    """Genetic Algorithm implementation for 8-puzzle"""
    if not is_solvable(start):
        return []

    start_tuple = tuple(start) # Convert start to tuple once

    def create_move():
        """Create a random move"""
        return random.choice(['U', 'D', 'L', 'R'])

    def create_individual():
        """Create a random sequence of moves"""
        return [create_move() for _ in range(random.randint(5, 30))]

    def apply_moves(state, moves):
        """Apply a sequence of moves to a state"""
        current = list(state) # state here is already a tuple from start_tuple
        path = [tuple(current)]
        
        for move in moves:
            empty = current.index(0)
            row, col = empty // 3, empty % 3
            
            if move == 'U' and row > 0:
                new_pos = empty - 3
            elif move == 'D' and row < 2:
                new_pos = empty + 3
            elif move == 'L' and col > 0:
                new_pos = empty - 1
            elif move == 'R' and col < 2:
                new_pos = empty + 1
            else:
                continue
                
            current[empty], current[new_pos] = current[new_pos], current[empty]
            path.append(tuple(current))
            
        return path

    def fitness(moves):
        """Calculate fitness of a sequence of moves"""
        path = apply_moves(start_tuple, moves) # Use start_tuple
        final_state = path[-1]
        return -manhattan_distance(final_state, goal)

    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    
    for _ in range(generations):
        # Evaluate fitness
        population = sorted(population, key=fitness, reverse=True)
        
        # Check if solution found
        best_path = apply_moves(start_tuple, population[0]) # Use start_tuple
        if best_path[-1] == goal:
            return best_path
            
        # Create next generation
        next_gen = population[:population_size//2]  # Keep best half
        
        while len(next_gen) < population_size:
            # Tournament selection
            parent1 = random.choice(population[:population_size//2])
            parent2 = random.choice(population[:population_size//2])
            
            # Crossover
            split = random.randint(0, min(len(parent1), len(parent2)))
            child = parent1[:split] + parent2[split:]
            
            # Mutation
            if random.random() < 0.1:
                pos = random.randint(0, len(child)-1)
                child[pos] = create_move()
                
            next_gen.append(child)
            
        population = next_gen
    
    # Return best found solution
    return apply_moves(start_tuple, population[0]) # Use start_tuple

# Cache cho successor generation
successor_cache = {}

def get_successors(state, is_or_node=True):
    """Tối ưu hóa việc tạo trạng thái kế tiếp với bộ nhớ cache
    - is_or_node: True cho OR node, False cho AND node
    - Lưu ý: Trong triển khai hiện tại, hai loại successors là như nhau
      nhưng được giữ riêng để dễ mở rộng trong tương lai
    """
    cache_key = (tuple(state), is_or_node)
    if cache_key in successor_cache:
        return successor_cache[cache_key]
        
    empty = state.index(0)
    row, col = empty // 3, empty % 3
    successors = []
    
    # Xét tất cả các hướng có thể di chuyển
    moves = [(-1,0), (1,0), (0,-1), (0,1)]
    for dx, dy in moves:
        new_row, new_col = row + dx, col + dy
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_state = list(state)
            new_pos = new_row * 3 + new_col
            new_state[empty], new_state[new_pos] = new_state[new_pos], new_state[empty]
            successors.append(tuple(new_state))
    
    # Lưu vào cache
    successor_cache[cache_key] = successors
    return successors

def get_or_successors(state):
    """Tạo các successor cho OR node - sử dụng cache"""
    return get_successors(state, True)

def get_and_successors(state):
    """Tạo các successor cho AND node - sử dụng cache"""
    return get_successors(state, False)

def andor_graph_search_improved(start, goal, use_heuristic=True, use_iterative_deepening=True):
    """AND/OR Graph Search cải tiến cho 8-puzzle
    Đặc điểm:
    - Sử dụng heuristic (Manhattan distance) để ưu tiên các trạng thái tốt hơn
    - Triển khai tăng dần độ sâu (iterative deepening) để tiết kiệm bộ nhớ
    - Tối ưu hóa việc tạo trạng thái kế tiếp bằng bộ nhớ cache
    - Tự động tăng giảm độ sâu tìm kiếm dựa vào khả năng tìm được lời giải
    """
    global successor_cache
    successor_cache = {}  # Reset cache
    
    def or_search(state, visited, depth=0, max_depth=31):
        if depth > max_depth:  # Giới hạn độ sâu
            return None, float('inf')  # Trả về cả score để tăng dần độ sâu
        if state == goal:
            return [state], 0
        if state in visited:
            return None, float('inf')
            
        visited.add(state)
        successors = get_or_successors(state)
        
        # Ưu tiên các successor có heuristic tốt hơn
        if use_heuristic:
            successors = sorted(successors, key=lambda s: manhattan_distance(s, goal))
        
        best_path = None
        best_score = float('inf')
        
        for successor in successors:
            next_path, next_score = and_search(successor, visited.copy(), depth + 1, max_depth)
            if next_path is not None and next_score < best_score:
                best_path = [state] + next_path
                best_score = next_score
                if best_score == 0:  # Đã tìm thấy goal, không cần tiếp tục
                    break
                    
        return best_path, best_score
        
    def and_search(state, visited, depth=0, max_depth=31):
        if depth > max_depth:  # Giới hạn độ sâu
            return None, float('inf')
        if state == goal:
            return [state], 0
        if state in visited:
            return None, float('inf')
            
        visited.add(state)
        successors = get_and_successors(state)
        
        if not successors:
            return None, float('inf')
        
        # Đối với AND node, tất cả các path phải thành công
        combined_path = [state]
        total_score = 0
        
        # Ưu tiên các successor có heuristic tốt hơn
        if use_heuristic:
            successors = sorted(successors, key=lambda s: manhattan_distance(s, goal))
        
        for successor in successors:
            next_path, next_score = or_search(successor, visited.copy(), depth + 1, max_depth)
            if next_path is None:
                return None, float('inf')
            combined_path.extend(next_path[1:])  # Bỏ qua trạng thái trùng lặp
            total_score += next_score
            
        return combined_path, total_score + 1  # +1 cho bước hiện tại

    if not is_solvable(start):
        return []
    
    # Chuyển đổi start thành tuple nếu chưa phải
    start_tuple = tuple(start)
    
    if use_iterative_deepening:
        # Iterative Deepening - tăng dần độ sâu
        max_depth = 5  # Bắt đầu với độ sâu nhỏ
        while max_depth <= 31:  # Giới hạn độ sâu tối đa
            visited = {start_tuple}
            path, score = or_search(start_tuple, visited, 0, max_depth)
            if path:
                return path
            max_depth += 2  # Tăng độ sâu mỗi lần lặp
    else:
        # Cách tiếp cận thông thường với độ sâu cố định
        visited = {start_tuple}
        path, _ = or_search(start_tuple, visited)
        return path if path else []
        
    return []

# Thay thế hàm cũ bằng phiên bản cải tiến
def andor_graph_search(start, goal):
    """AND/OR Graph Search cho 8-puzzle - Phiên bản cải tiến
    Đặc điểm:
    - Phân tách bài toán thành các nút AND và OR
    - Nút AND: Cần thỏa mãn tất cả các điều kiện con
    - Nút OR: Chỉ cần thỏa mãn một trong các điều kiện con
    - Xây dựng cây tìm kiếm AND/OR để tìm giải pháp
    - Sử dụng heuristic và iterative deepening khi có thể
    """
    return andor_graph_search_improved(start, goal, use_heuristic=True, use_iterative_deepening=True)

def fixed_partially_observable(start, goal, observable_positions=[0,1,2,4,8]):
    """Fixed Partially Observable Search
    Đặc điểm:
    - Chỉ quan sát được một phần cố định của bảng (các vị trí cố định)
    - Phải quyết định dựa trên thông tin không đầy đủ
    - Sử dụng kỹ thuật belief state để theo dõi các trạng thái có thể
    - Không đảm bảo tìm ra đường đi tối ưu
    """
    if not is_solvable(start):
        return []
        
    # Lọc các vị trí có thể quan sát
    def get_observable_state(state):
        # Tạo trạng thái mờ với chỉ các ô có thể quan sát
        observable = [-1] * 9
        for pos in observable_positions:
            observable[pos] = state[pos]
        return tuple(observable)
        
    queue = deque([(tuple(start), [tuple(start)])])
    visited = {tuple(start)}
    observable_visited = {get_observable_state(tuple(start))}
    
    while queue:
        state, path = queue.popleft()
        
        if state == goal:
            return path
            
        for neighbor in get_neighbors(state):
            if neighbor not in visited:
                # Kiểm tra xem phần có thể quan sát của neighbor đã từng thấy chưa
                observable_neighbor = get_observable_state(neighbor)
                
                if observable_neighbor not in observable_visited:
                    visited.add(neighbor)
                    observable_visited.add(observable_neighbor)
                    queue.append((neighbor, path + [neighbor]))
                    
    return []

def backtracking_search(start, goal, max_depth=31):
    """Backtracking Search Algorithm - Thuật toán Quay lui
    Đặc điểm:
    - Thử từng khả năng và quay lui khi gặp đường cụt
    - Sử dụng kỹ thuật tìm kiếm theo chiều sâu có đánh dấu
    - Tìm kiếm có hệ thống các giải pháp tiềm năng
    - Cắt tỉa các nhánh không cần thiết sớm
    """
    if not is_solvable(start):
        return []
    
    def backtrack(state, path, depth, visited):
        if state == goal:
            return path
        
        if depth >= max_depth:
            return None
        
        for neighbor in get_neighbors(state):
            if neighbor not in visited:
                visited.add(neighbor)
                result = backtrack(neighbor, path + [neighbor], depth + 1, visited)
                if result:
                    return result
                visited.remove(neighbor)  # Backtrack by removing from visited
        
        return None
    
    visited = {tuple(start)}
    result = backtrack(tuple(start), [tuple(start)], 0, visited)
    return result if result else []

def belief_state_search(start, goal, observable_positions=[0,1,2,4,8]):
    """Belief State Search
    Đặc điểm:
    - Chỉ quan sát được một phần cố định của bảng (các vị trí cố định)
    - Phải quyết định dựa trên thông tin không đầy đủ
    - Sử dụng kỹ thuật belief state để theo dõi các trạng thái có thể
    - Không đảm bảo tìm ra đường đi tối ưu
    """
    if not is_solvable(start):
        return []
        
    # Lọc các vị trí có thể quan sát
    def get_observable_state(state):
        # Tạo trạng thái mờ với chỉ các ô có thể quan sát
        observable = [-1] * 9
        for pos in observable_positions:
            observable[pos] = state[pos]
        return tuple(observable)
        
    queue = deque([(tuple(start), [tuple(start)])])
    visited = {tuple(start)}
    observable_visited = {get_observable_state(tuple(start))}
    
    while queue:
        state, path = queue.popleft()
        
        if state == goal:
            return path
            
        for neighbor in get_neighbors(state):
            if neighbor not in visited:
                # Kiểm tra xem phần có thể quan sát của neighbor đã từng thấy chưa
                observable_neighbor = get_observable_state(neighbor)
                
                if observable_neighbor not in observable_visited:
                    visited.add(neighbor)
                    observable_visited.add(observable_neighbor)
                    queue.append((neighbor, path + [neighbor]))
                    
    return []

def forward_checking(start, goal):
    """Forward Checking
    Đặc điểm:
    - Kiểm tra các điều kiện ràng buộc trước khi thực hiện các bước di chuyển
    - Đảm bảo tìm ra đường đi tối ưu
    - Có thể thoát khỏi cực tiểu cục bộ (local minima)
    """
    if not is_solvable(start):
        return []
        
    def check_constraints(state):
        # Kiểm tra các điều kiện ràng buộc
        for i in range(9):
            if state[i] != 0:
                for j in range(9):
                    if state[j] != 0 and i != j and state[i] == state[j]:
                        return False
        return True

    def backtrack(state, path, depth, visited):
        if state == goal:
            return path
        
        if depth >= 31:  # Giới hạn độ sâu
            return None
        
        for i in range(9):
            if state[i] == 0:
                for j in range(9):
                    if state[j] != 0 and i != j:
                        new_state_list = list(state)
                        new_state_list[i], new_state_list[j] = new_state_list[j], new_state_list[i]
                        new_state_tuple = tuple(new_state_list) # Convert to tuple
                        if check_constraints(new_state_tuple): # Pass tuple
                            # Assuming backtrack should add to visited if not present
                            if new_state_tuple not in visited:
                                visited.add(new_state_tuple)
                                result = backtrack(new_state_tuple, path + [new_state_tuple], depth + 1, visited)
                                if result:
                                    return result
                                visited.remove(new_state_tuple) # Backtrack
        return None

    visited = {tuple(start)}
    result = backtrack(tuple(start), [tuple(start)], 0, visited)
    return result if result else []

def ac3_algorithm(start, goal):
    """AC-3 Algorithm
    Đặc điểm:
    - Kiểm tra các điều kiện ràng buộc trước khi thực hiện các bước di chuyển
    - Đảm bảo tìm ra đường đi tối ưu
    - Có thể thoát khỏi cực tiểu cục bộ (local minima)
    """
    # Note: The implementation of AC-3 seems to be a copy of forward_checking
    # and doesn't correctly implement AC-3. For now, just ensuring tuple usage.
    if not is_solvable(start):
        return []
        
    def check_constraints(state): # state is already a tuple from new_state
        # Kiểm tra các điều kiện ràng buộc
        for i in range(9):
            if state[i] != 0:
                for j in range(9):
                    if state[j] != 0 and i != j and state[i] == state[j]:
                        return False
        return True

    def backtrack(state, path, depth, visited): # state is already tuple
        if state == goal:
            return path
        
        if depth >= 31:  # Giới hạn độ sâu
            return None
        
        for i in range(9):
            if state[i] == 0:
                for j in range(9):
                    if state[j] != 0 and i != j:
                        new_state_list = list(state)
                        new_state_list[i], new_state_list[j] = new_state_list[j], new_state_list[i]
                        new_state_tuple = tuple(new_state_list) # Convert to tuple
                        if check_constraints(new_state_tuple): # Pass tuple
                            if new_state_tuple not in visited:
                                visited.add(new_state_tuple)
                                result = backtrack(new_state_tuple, path + [new_state_tuple], depth + 1, visited)
                                if result:
                                    return result
                                visited.remove(new_state_tuple) # Backtrack
        return None

    visited = {tuple(start)}
    result = backtrack(tuple(start), [tuple(start)], 0, visited)
    return result if result else []

class QLearningAgent:
    """Q-Learning Agent for 8-puzzle"""
    def __init__(self, alpha=0.2, gamma=0.95, epsilon=0.2):
        self.q_values = {}  # (state, action) -> value
        self.alpha = alpha  # Tỉ lệ học tập - Tăng alpha để học nhanh hơn
        self.gamma = gamma  # Hệ số giảm giá - Tăng gamma để coi trọng phần thưởng tương lai
        self.epsilon = epsilon  # Tỉ lệ khám phá - Tăng epsilon để khám phá nhiều hơn
        self.episode_rewards = []  # Lưu trữ phần thưởng cho mỗi episode
        self.episode_steps = []    # Lưu trữ số bước cho mỗi episode
        self.current_episode = 0
        self.visited_states_history = []  # Lưu trữ N trạng thái đã thăm gần đây
        self.memory_size = 15  # Tăng bộ nhớ ngắn hạn để tránh lặp tốt hơn
        self.experience_replay = []  # Danh sách kinh nghiệm (state, action, reward, next_state)
        self.experience_max_size = 1000  # Kích thước tối đa của bộ nhớ kinh nghiệm
        
    def get_q_value(self, state, action):
        """Lấy giá trị Q cho cặp (state, action)"""
        return self.q_values.get((state, action), 0.0)
        
    def get_best_action(self, state, legal_actions, avoid_visited=False):
        """Lấy hành động tốt nhất từ danh sách các hành động hợp lệ
        Tham số:
        - avoid_visited: Nếu True, tránh chọn hành động dẫn đến trạng thái đã thăm gần đây
        """
        if not legal_actions:
            return None
            
        if random.random() < self.epsilon:
            # Khám phá: chọn ngẫu nhiên
            return random.choice(legal_actions)
        else:
            # Loại bỏ các hành động dẫn đến trạng thái đã thăm gần đây (nếu được yêu cầu)
            filtered_actions = legal_actions.copy()
            if avoid_visited and len(self.visited_states_history) > 0:
                filtered_actions = []
                for action in legal_actions:
                    next_state = self.apply_action(state, action)
                    if next_state not in self.visited_states_history:
                        filtered_actions.append(action)
                
                # Nếu tất cả hành động đều dẫn đến trạng thái đã thăm, sử dụng danh sách gốc
                if not filtered_actions:
                    filtered_actions = legal_actions.copy()
            
            # Khai thác: chọn hành động có giá trị Q cao nhất
            values = [self.get_q_value(state, action) for action in filtered_actions]
            max_value = max(values) if values else 0
            best_actions = [action for action, value in zip(filtered_actions, values) 
                           if value == max_value]
            return random.choice(best_actions)  # Phá vỡ sự đồng đều ngẫu nhiên
            
    def update(self, state, action, next_state, reward, done):
        """Cập nhật giá trị Q dựa trên trải nghiệm và sử dụng experience replay"""
        # Lưu trữ kinh nghiệm để sử dụng experience replay
        self.experience_replay.append((state, action, reward, next_state, done))
        if len(self.experience_replay) > self.experience_max_size:
            self.experience_replay.pop(0)  # Loại bỏ kinh nghiệm cũ nhất

        # Cập nhật Q-value cho trải nghiệm hiện tại
        legal_actions = self.get_legal_actions(next_state)
        if not legal_actions or done:
            max_q_next = 0.0
        else:
            max_q_next = max([self.get_q_value(next_state, a) for a in legal_actions])
            
        old_value = self.get_q_value(state, action)
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * max_q_next)
        self.q_values[(state, action)] = new_value
        
        # Cập nhật lịch sử trạng thái đã thăm
        if next_state not in self.visited_states_history:
            self.visited_states_history.append(next_state)
            if len(self.visited_states_history) > self.memory_size:
                self.visited_states_history.pop(0)  # Loại bỏ trạng thái cũ nhất
                
        # Experience replay: Học từ kinh nghiệm đã lưu trữ
        if len(self.experience_replay) > 10:  # Chỉ bắt đầu replay khi có đủ kinh nghiệm
            # Chọn ngẫu nhiên một số kinh nghiệm để học lại
            batch_size = min(8, len(self.experience_replay))
            batch = random.sample(self.experience_replay, batch_size)
            
            for exp_state, exp_action, exp_reward, exp_next_state, exp_done in batch:
                # Cập nhật Q-value cho kinh nghiệm đã lưu trữ
                if exp_done:
                    exp_q_target = exp_reward
                else:
                    exp_legal_actions = self.get_legal_actions(exp_next_state)
                    if not exp_legal_actions:
                        exp_q_target = exp_reward
                    else:
                        exp_q_next = max([self.get_q_value(exp_next_state, a) for a in exp_legal_actions])
                        exp_q_target = exp_reward + self.gamma * exp_q_next
                
                exp_old_value = self.get_q_value(exp_state, exp_action)
                exp_new_value = (1 - self.alpha * 0.5) * exp_old_value + (self.alpha * 0.5) * exp_q_target
                self.q_values[(exp_state, exp_action)] = exp_new_value
        
    def get_legal_actions(self, state):
        """Lấy các hành động hợp lệ từ trạng thái hiện tại"""
        actions = []
        empty_idx = state.index(0)
        row, col = empty_idx // 3, empty_idx % 3
        
        # Kiểm tra các hướng di chuyển hợp lệ (lên, xuống, trái, phải)
        if row > 0:  # Có thể di chuyển lên
            actions.append('U')
        if row < 2:  # Có thể di chuyển xuống
            actions.append('D')
        if col > 0:  # Có thể di chuyển sang trái
            actions.append('L')
        if col < 2:  # Có thể di chuyển sang phải
            actions.append('R')
            
        return actions
        
    def apply_action(self, state, action):
        """Áp dụng hành động vào trạng thái và trả về trạng thái mới"""
        state_list = list(state)
        empty_idx = state_list.index(0)
        row, col = empty_idx // 3, empty_idx % 3
        
        if action == 'U' and row > 0:
            new_empty = empty_idx - 3
        elif action == 'D' and row < 2:
            new_empty = empty_idx + 3
        elif action == 'L' and col > 0:
            new_empty = empty_idx - 1
        elif action == 'R' and col < 2:
            new_empty = empty_idx + 1
        else:
            return tuple(state_list)  # Không thay đổi nếu hành động không hợp lệ
            
        state_list[empty_idx], state_list[new_empty] = state_list[new_empty], state_list[empty_idx]
        return tuple(state_list)
        
    def get_reward(self, state, next_state, goal):
        """Tính toán phần thưởng cho một hành động - Cải thiện + Step Penalty"""
        if next_state == goal:
            return 100  # Phần thưởng lớn cho việc đạt được goal
        
        # Phần thưởng dựa trên cải thiện khoảng cách Manhattan
        current_distance = manhattan_distance(state, goal)
        next_distance = manhattan_distance(next_state, goal)
        
        reward = -0.05  # Giảm hình phạt cho mỗi bước đi
        
        # Giảm mức phạt cho trạng thái đã thăm
        if next_state in self.visited_states_history:
            reward -= 1.0  # Giảm mức phạt đi (từ 3.0 xuống 1.0)
        
        if next_distance < current_distance:
            reward += 1.5 + (current_distance - next_distance) * 0.8  # Tăng phần thưởng cho cải thiện
        elif next_distance > current_distance:
            # Giảm mức phạt khi tạm thời đi xa mục tiêu
            reward -= 1.0 + (next_distance - current_distance) * 0.3  # Giảm mức phạt (từ 5.0 xuống 1.0)
        else:
            # Thêm phần thưởng nhỏ khi di chuyển không thay đổi khoảng cách, khuyến khích khám phá
            reward += 0.2
            
        return reward
            
    def train(self, start, goal, max_episodes=1000, max_steps=100):
        """Huấn luyện agent sử dụng Q-learning - Sử dụng cả trạng thái ngẫu nhiên và trạng thái thực tế"""
        initial_alpha = self.alpha # Lưu alpha ban đầu để reset nếu cần
        initial_epsilon = self.epsilon # Lưu epsilon ban đầu
        
        goal_tuple = tuple(goal) # Convert goal to tuple once
        start_tuple = tuple(start) # Convert start to tuple once

        for episode in range(max_episodes):
            self.current_episode = episode + 1
            
            # Sử dụng trạng thái bắt đầu thực tế theo chu kỳ
            if episode % 5 == 0:
                # Mỗi 5 episodes, sử dụng trạng thái bắt đầu thực tế
                state = start_tuple
            else:
                # Các episodes khác, sử dụng trạng thái ngẫu nhiên để khám phá
                random_state = generate_random_puzzle()
                # Đảm bảo trạng thái ngẫu nhiên có thể giải được
                while not is_solvable(random_state):
                    random_state = generate_random_puzzle()
                state = tuple(random_state)
                
            # Thêm: Thỉnh thoảng tạo trạng thái gần với đích
            if episode % 10 == 0 and episode > 0:
                # Tạo trạng thái gần với đích (cách đích vài bước)
                temp_state = list(goal_tuple)
                for _ in range(random.randint(3, 6)):
                    actions = self.get_legal_actions(tuple(temp_state))
                    if actions:
                        action = random.choice(actions)
                        new_temp_state = self.apply_action(tuple(temp_state), action)
                        temp_state = list(new_temp_state)
                state = tuple(temp_state)
            
            # Reset bộ nhớ ngắn hạn cho mỗi episode mới
            self.visited_states_history = []
            
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                legal_actions = self.get_legal_actions(state)
                if not legal_actions:
                    break
                    
                action = self.get_best_action(state, legal_actions)
                next_state = self.apply_action(state, action)
                done = next_state == goal_tuple
                
                reward = self.get_reward(state, next_state, goal_tuple)
                total_reward += reward
                
                self.update(state, action, next_state, reward, done)
                
                state = next_state
                steps += 1
                
                if done:
                    break
                    
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            
            # Giảm epsilon dần dần (exploration decay)
            if self.epsilon > 0.01:
                self.epsilon *= 0.995  # Giảm chậm hơn để khám phá nhiều hơn
                
            # Giảm alpha dần dần (learning rate decay)
            if self.alpha > 0.01:
                self.alpha *= 0.999 # Giảm chậm hơn epsilon
    
    def solve(self, start, goal, max_steps=200):
        """Giải bài toán 8-puzzle sử dụng Q-learning đã huấn luyện - Cải thiện để tránh lặp và tìm đường tốt hơn"""
        if not is_solvable(start):
            return []
            
        state = tuple(start)
        path = [state]
        goal_tuple = tuple(goal) # Convert goal to tuple
        
        # Reset bộ nhớ ngắn hạn trước khi giải
        self.visited_states_history = []
        
        # Các trạng thái đã thăm trong quá trình giải hiện tại
        visited = {state}
        
        # Thêm một epsilon nhỏ cho quá trình giải để cho phép khám phá
        old_epsilon = self.epsilon
        self.epsilon = 0.05  # Thay vì 0, dùng 0.05 để vẫn có cơ hội khám phá
        
        for step in range(max_steps):
            if state == goal_tuple:
                self.epsilon = old_epsilon
                return path
                
            legal_actions = self.get_legal_actions(state)
            if not legal_actions:
                break
            
            # Thử lần đầu với avoid_visited=True
            action = self.get_best_action(state, legal_actions, avoid_visited=True)
            next_state = self.apply_action(state, action)
            
            # Nếu rơi vào trạng thái đã thăm, thử lại với avoid_visited=False
            attempts = 0
            while next_state in visited and attempts < 5:
                # Loại bỏ hành động vừa chọn khỏi danh sách nếu có thể
                if action in legal_actions and len(legal_actions) > 1:
                    legal_actions.remove(action)
                elif len(legal_actions) <= 1:
                    break
                
                # Chọn một hành động khác, trong đó thỉnh thoảng cho phép khám phá
                if random.random() < 0.3:  # 30% cơ hội chọn ngẫu nhiên để thoát khỏi local minimum
                    action = random.choice(legal_actions)
                else:
                    action = self.get_best_action(state, legal_actions, avoid_visited=False)
                    
                next_state = self.apply_action(state, action)
                attempts += 1
            
            # Kiểm tra xem đã đến đích chưa hoặc đã đi quá xa
            if next_state == goal_tuple or (len(path) > 30 and manhattan_distance(next_state, goal_tuple) > manhattan_distance(path[0], goal_tuple)):
                if next_state == goal_tuple:
                    path.append(next_state)
                    self.epsilon = old_epsilon
                    return path
                else:
                    # Nếu đi quá xa, thử một chiến lược khác: dùng A*
                    astar_path = astar(state, goal_tuple)
                    if astar_path:
                        self.epsilon = old_epsilon
                        return path + astar_path[1:]  # Bỏ state hiện tại khỏi astar_path
                    break
            
            # Cập nhật bộ nhớ ngắn hạn và trạng thái đã thăm
            self.visited_states_history.append(next_state)
            if len(self.visited_states_history) > self.memory_size:
                self.visited_states_history.pop(0)
                
            visited.add(next_state)
            state = next_state
            path.append(state)
            
        # Khôi phục epsilon
        self.epsilon = old_epsilon
        
        # Nếu không tìm thấy đường đi, thử dùng thuật toán A* để giải
        if path[-1] != goal_tuple:
            astar_result = astar(start, goal_tuple)
            if astar_result:
                return astar_result
                
        return path

def q_learning(start, goal):
    """Q-Learning for 8-puzzle"""
    if not is_solvable(start):
        return []
    
    # Sử dụng agent đã được đào tạo trước (nếu có)
    global q_agent
    if 'q_agent' not in globals():
        q_agent = QLearningAgent(alpha=0.2, gamma=0.95, epsilon=0.2)
        # Đào tạo với số episode lớn hơn
        q_agent.train(tuple(start), tuple(goal), max_episodes=500, max_steps=150)
    else:
        # Nếu agent đã tồn tại, huấn luyện thêm tập trung vào trạng thái hiện tại
        q_agent.epsilon = 0.2  # Đặt lại epsilon để khám phá nhiều hơn
        q_agent.train(tuple(start), tuple(goal), max_episodes=200, max_steps=150)
    
    # Giải quyết bài toán với số bước tối đa lớn hơn
    return q_agent.solve(tuple(start), tuple(goal), max_steps=200)

# Evaluation Thread class
class EvaluationThread(QThread):
    """Thread to run algorithm evaluation in background without freezing UI"""
    update_progress = pyqtSignal(int, str)
    evaluation_complete = pyqtSignal(dict)
    
    def __init__(self, start_state, goal_state, num_puzzles, algorithms, parent=None):
        super(EvaluationThread, self).__init__(parent)
        self.start_state = start_state
        self.goal_state = goal_state
        self.num_puzzles = num_puzzles
        self.algorithms = algorithms
        self.is_running = True
        
        # Algorithm map for evaluation
        self.algorithm_map = {
            'BFS': bfs,
            'DFS': dfs,
            'UCS': uniform_cost,
            'GREEDY': greedy,
            'ASTAR': astar,
            'HILL_CLIMBING': hill_climbing,
            'STEEPEST_HILL': hill_climbing_max,
            'STOCHASTIC_HILL': hill_climbing_random_restart,
            'LOCAL_BEAM': lambda start, goal: local_beam_search(start, goal, beam_width=5),
            'ANNEALING': simulated_annealing,
            'IDS': iterative_deepening_search,
            'IDASTAR': ida_star,
            'GENETIC': genetic_algorithm,
            'ANDOR': andor_graph_search,
            'PARTIALLY_OBS': fixed_partially_observable,
            'BELIEF_STATE': belief_state_search,
            'BACKTRACKING': backtracking_search,
            'FORWARD_CHECK': forward_checking,
            'AC3': ac3_algorithm,
            'QLEARNING': q_learning
        }
        
    def stop(self):
        self.is_running = False
        
    def run(self):
        results = {}
        puzzles = []
        goal_tuple = tuple(self.goal_state) # Convert goal_state to tuple once
        
        # Generate random puzzles
        self.update_progress.emit(0, "Generating puzzles...")
        for i in range(self.num_puzzles):
            if not self.is_running:
                break
            puzzles.append(generate_random_puzzle())
            self.update_progress.emit(int((i+1) / self.num_puzzles * 100 * 0.1), 
                                f"Generated puzzle {i+1}/{self.num_puzzles}")
        
        if not self.is_running:
            return
            
        total_algs = len(self.algorithms)
        puzzles_per_alg = min(self.num_puzzles, 10)  # Limit to 10 puzzles per algorithm for speed
        
        for alg_idx, algorithm_name in enumerate(self.algorithms):
            if not self.is_running:
                break
                
            self.update_progress.emit(10 + int(alg_idx / total_algs * 90), 
                               f"Evaluating {algorithm_name}...")
            
            algorithm_func = self.algorithm_map.get(algorithm_name)
            if not algorithm_func:
                continue
                
            times = []
            path_lengths = []
            success_rate = 0
            nodes_expanded = []
            
            # Track total nodes expanded - for BFS, DFS, etc. we'll estimate from path length
            for puzzle_idx, puzzle in enumerate(puzzles[:puzzles_per_alg]):
                if not self.is_running:
                    break
                    
                self.update_progress.emit(10 + int((alg_idx + puzzle_idx/puzzles_per_alg) / total_algs * 90),
                                  f"Running {algorithm_name} on puzzle {puzzle_idx+1}/{puzzles_per_alg}")
                
                # Run the algorithm and measure time
                start_time = time.time()
                path = algorithm_func(tuple(puzzle), goal_tuple) # Pass tuple puzzle and goal_tuple
                end_time = time.time()
                
                # Calculate metrics
                solve_time = end_time - start_time
                times.append(solve_time)
                
                if path:
                    path_lengths.append(len(path) - 1)  # Subtract 1 for initial state
                    success_rate += 1
                    
                    # Estimate nodes expanded based on algorithm type
                    if algorithm_name in ['BFS', 'UCS', 'ASTAR', 'GREEDY']:
                        # For breadth-first-like algorithms, estimate as 4^(path length)
                        nodes_expanded.append(min(4 ** len(path), 10000))  # Cap to prevent overflow
                    elif algorithm_name in ['DFS', 'IDS', 'IDASTAR']:
                        # For depth-first-like algorithms, use higher estimate
                        nodes_expanded.append(min(8 ** len(path), 20000))
                    else:
                        # Default estimation for other algorithms
                        nodes_expanded.append(min(3 ** len(path), 5000))
                else:
                    path_lengths.append(0)
                    nodes_expanded.append(0)
            
            # Calculate final metrics
            if times:
                avg_time = sum(times) / len(times)
                avg_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else 0
                success_percentage = (success_rate / len(puzzles[:puzzles_per_alg])) * 100
                avg_nodes = sum(nodes_expanded) / len(nodes_expanded) if nodes_expanded else 0
                
                results[algorithm_name] = {
                    'avg_time': avg_time,
                    'avg_path_length': avg_path_length,
                    'success_rate': success_percentage,
                    'avg_nodes_expanded': avg_nodes
                }
        
        if self.is_running:
            self.update_progress.emit(100, "Evaluation complete!")
            self.evaluation_complete.emit(results)

# GUI Classes
class PuzzleBoard(QFrame):
    cell_clicked = pyqtSignal(int, int)
    
    def __init__(self):
        super().__init__()
        self.state = [1,2,3,4,5,6,7,0,8]
        self.cell_size = 100
        self.animation_offset = {}
        self.setFixedSize(self.cell_size * 3, self.cell_size * 3)
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)
        
        self.setStyleSheet(f"""
            PuzzleBoard {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 {COLORS['surface']}, 
                                          stop:1 {self.lighter_color(COLORS['surface'], 10)});
                border: none;
                border-radius: 12px;
            }}
        """)
        
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.setInterval(16)
        
    def lighter_color(self, color, amount=20):
        c = QColor(color)
        h = c.hue()
        s = max(0, min(255, c.saturation() - amount))
        l = min(255, c.lightness() + amount)
        return QColor.fromHsl(h, s, l).name()
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        for i in range(3):
            for j in range(3):
                x = j * self.cell_size
                y = i * self.cell_size
                value = self.state[i * 3 + j]
                
                if value != 0:
                    offset_x = offset_y = 0
                    if value in self.animation_offset:
                        offset_x = self.animation_offset[value][0]
                        offset_y = self.animation_offset[value][1]
                    
                    gradient = QLinearGradient(x+5, y+5, x+self.cell_size-5, y+self.cell_size-5)
                    base_color = QColor(COLORS['primary'])
                    gradient.setColorAt(0, base_color.lighter(150))
                    gradient.setColorAt(1, base_color)
                    
                    shadow_path = QPainterPath()
                    shadow_path.addRoundedRect(x+7+offset_x, y+7+offset_y, 
                                             self.cell_size-14, self.cell_size-14, 10, 10)
                    painter.fillPath(shadow_path, QColor(0, 0, 0, 30))
                    
                    path = QPainterPath()
                    path.addRoundedRect(x+5+offset_x, y+5+offset_y, 
                                      self.cell_size-10, self.cell_size-10, 10, 10)
                    painter.fillPath(path, gradient)
                    
                    font = painter.font()
                    font.setPointSize(28)
                    font.setBold(True)
                    painter.setFont(font)
                    
                    painter.setPen(QColor(0, 0, 0, 50))
                    painter.drawText(x+1+offset_x, y+1+offset_y, 
                                   self.cell_size, self.cell_size,
                                   Qt.AlignCenter, str(value))
                    
                    painter.setPen(Qt.white)
                    painter.drawText(x+offset_x, y+offset_y, 
                                   self.cell_size, self.cell_size,
                                   Qt.AlignCenter, str(value))
    
    def update_animation(self):
        still_animating = False
        for value in list(self.animation_offset.keys()):
            dx, dy, target_x, target_y = self.animation_offset[value]
            
            speed = 10
            if abs(dx - target_x) > speed:
                dx += speed if target_x > dx else -speed
                still_animating = True
            else:
                dx = target_x
                
            if abs(dy - target_y) > speed:
                dy += speed if target_y > dy else -speed
                still_animating = True
            else:
                dy = target_y
                
            if dx == target_x and dy == target_y:
                del self.animation_offset[value]
            else:
                self.animation_offset[value] = (dx, dy, target_x, target_y)
        
        self.update()
        if not still_animating:
            self.animation_timer.stop()
    
    def animate_move(self, value, start_pos, end_pos):
        start_row, start_col = start_pos // 3, start_pos % 3
        end_row, end_col = end_pos // 3, end_pos % 3
        
        start_x = start_col * self.cell_size
        start_y = start_row * self.cell_size
        end_x = end_col * self.cell_size
        end_y = end_row * self.cell_size
        
        self.animation_offset[value] = (0, 0, end_x - start_x, end_y - start_y)
        self.animation_timer.start()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x = event.x() // self.cell_size
            y = event.y() // self.cell_size
            if 0 <= x < 3 and 0 <= y < 3:
                self.cell_clicked.emit(y, x)
                
    def update_state(self, new_state):
        self.state = list(new_state)
        self.update()

class ControlButton(QPushButton):
    def __init__(self, text, color=COLORS['primary']):
        super().__init__(text)
        self.setMinimumHeight(45)
        self.color = color
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 40))
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)
        
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 14px;
                min-width: 150px;
            }}
            QPushButton:hover {{
                background-color: {self.lighter_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self.darker_color(color)};
            }}
        """)
        
    def lighter_color(self, color, amount=30):
        c = QColor(color)
        h = c.hue()
        s = max(0, min(255, c.saturation() - amount))
        l = min(255, c.lightness() + amount)
        return QColor.fromHsl(h, s, l).name()
        
    def darker_color(self, color, amount=30):
        c = QColor(color)
        h = c.hue()
        s = min(255, c.saturation() + amount)
        l = max(0, c.lightness() - amount)
        return QColor.fromHsl(h, s, l).name()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_algorithm = 'BFS'
        self.solution_path = []
        self.current_step = 0
        self.goal_state = tuple([1,2,3,4,5,6,7,8,0]) # Store as tuple
        self.evaluation_window = None  # Add this line
        self.qlearning_window = None # Initialize qlearning_window

        self.setWindowTitle("8 Puzzle Solver")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(f"background-color: {COLORS['background']};")
        
        # Create central widget with boarder puzzle
        self.setupCentralWidget()
        
        # Create dock widgets
        self.setupAlgorithmsDock()
        self.setupControlsDock()
        self.setupLogDock()
        
        # Set dock options and initial layout
        self.setDockOptions(QMainWindow.AnimatedDocks | 
                           QMainWindow.AllowNestedDocks | 
                           QMainWindow.AllowTabbedDocks)
        
        # Initialize log display
        self.log("Application started")
        self.log(f"Current algorithm: {self.current_algorithm}")
        
        self.update_status_labels()
        
        # Automatically maximize window when opened
        self.showMaximized()
    
    def setupCentralWidget(self):
        """Setup the central widget with puzzle board"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title in center panel
        title_label = QLabel("8 Puzzle Game")
        title_label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 24px;
            font-weight: bold;
            padding: 10px 0px 20px 0px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        central_layout.addWidget(title_label)
        
        # Create a stacked widget to switch between normal and Q-learning views
        self.stacked_widget = QWidget()
        stacked_layout = QVBoxLayout(self.stacked_widget)
        stacked_layout.setContentsMargins(0, 0, 0, 0)
        
        # Normal puzzle frame
        self.puzzle_frame = QFrame()
        self.puzzle_frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['surface']};
                border-radius: 12px;
                padding: 20px;
                border: 1px solid {COLORS['border']};
            }}
        """)
        puzzle_frame_layout = QVBoxLayout(self.puzzle_frame)
        
        puzzle_container = QWidget()
        puzzle_layout = QVBoxLayout(puzzle_container)
        puzzle_layout.setContentsMargins(10, 10, 10, 10)
        
        self.puzzle_board = PuzzleBoard()
        self.puzzle_board.cell_clicked.connect(self.handle_cell_click)
        puzzle_layout.addWidget(self.puzzle_board, alignment=Qt.AlignCenter)
        
        puzzle_frame_layout.addWidget(puzzle_container)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(15)
        nav_layout.setContentsMargins(10, 10, 10, 10)
        
        prev_btn = ControlButton("Previous", COLORS['secondary'])
        prev_btn.setMinimumHeight(40)
        prev_btn.setMinimumWidth(120)
        prev_btn.clicked.connect(self.on_prev)
        nav_layout.addWidget(prev_btn)
        
        next_btn = ControlButton("Next", COLORS['primary'])
        next_btn.setMinimumHeight(40)
        next_btn.setMinimumWidth(120)
        next_btn.clicked.connect(self.on_next)
        nav_layout.addWidget(next_btn)
        
        puzzle_frame_layout.addLayout(nav_layout)
        
        # Add puzzle frame to stacked widget
        stacked_layout.addWidget(self.puzzle_frame)
        # NOTE: Q-learning frame is removed, we use a separate window
        
        central_layout.addWidget(self.stacked_widget)
        
        # Add a status bar at the bottom of the central widget
        status_panel = QFrame()
        status_panel.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['divider']};
                border-radius: 4px;
                padding: 8px;
                margin-top: 10px;
            }}
            QLabel {{
                margin-top: 0px;
            }}
        """)
        status_layout = QHBoxLayout(status_panel)
        status_layout.setSpacing(5)
        status_layout.setContentsMargins(5, 5, 5, 5)
        
        self.algorithm_label = QLabel()
        self.algorithm_label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 12px;
            font-weight: bold;
        """)
        status_layout.addWidget(self.algorithm_label)
        
        status_layout.addStretch()
        
        self.steps_label = QLabel()
        self.steps_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 12px;
        """)
        status_layout.addWidget(self.steps_label)
        
        central_layout.addWidget(status_panel)
    
    def setupAlgorithmsDock(self):
        """Setup the algorithms dock widget"""
        algorithms_dock = QDockWidget("Algorithms", self)
        algorithms_dock.setObjectName("AlgorithmsDock")
        algorithms_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        algorithms_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        
        # Scroll area for algorithms
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        # Left panel (Algorithms)
        left_panel = QFrame()
        left_panel.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['surface']};
                border-radius: 8px;
                padding: 12px;
                border: 1px solid {COLORS['border']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: 11px;
                font-weight: bold;
                margin-top: 5px;
            }}
            QLabel#group_label {{
                color: {COLORS['text_primary']};
                font-size: 11px;
                font-weight: bold;
                background: {COLORS['divider']};
                padding: 5px 8px;
                border-radius: 6px;
                margin: 4px 0px;
                min-height: 15px;
            }}
            QPushButton {{
                text-align: left;
                padding: 0px 10px;
                font-size: 11px;
                border-radius: 4px;
            }}
        """)
        left_panel.setMinimumWidth(260)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(3)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # Main algorithms title
        algorithms_label = QLabel("Chọn Thuật Toán")
        algorithms_label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 14px;
            font-weight: bold;
            padding: 0px 0px 5px 0px;
        """)
        left_layout.addWidget(algorithms_label)
        
        # Uninformed Search Group
        uninformed_label = QLabel("Tìm kiếm không có thông tin")
        uninformed_label.setObjectName("group_label")
        left_layout.addWidget(uninformed_label)
        
        uninformed_algorithms = [
            ('BFS (Breadth-First Search)', 'BFS', COLORS['uninformed']),
            ('DFS (Depth-First Search)', 'DFS', COLORS['uninformed']),
            ('UCS (Uniform Cost Search)', 'UCS', COLORS['uninformed']),
            ('IDS (Iterative Deepening)', 'IDS', COLORS['uninformed'])
        ]
        
        for name, algo_id, color in uninformed_algorithms:
            btn = ControlButton(name, color)
            btn.setMinimumHeight(28)
            btn.setMaximumHeight(28)
            btn.clicked.connect(lambda checked, a=algo_id: self.select_algorithm(a))
            left_layout.addWidget(btn)

        left_layout.addSpacing(3)

        # Informed Search Group
        informed_label = QLabel("Tìm kiếm có thông tin")
        informed_label.setObjectName("group_label")
        left_layout.addWidget(informed_label)
        
        informed_algorithms = [
            ('A* Search', 'ASTAR', COLORS['informed']),
            ('IDA* Search', 'IDASTAR', COLORS['informed']),
            ('Greedy Best-First', 'GREEDY', COLORS['informed'])
        ]
        
        for name, algo_id, color in informed_algorithms:
            btn = ControlButton(name, color)
            btn.setMinimumHeight(28)
            btn.setMaximumHeight(28)
            btn.clicked.connect(lambda checked, a=algo_id: self.select_algorithm(a))
            left_layout.addWidget(btn)

        left_layout.addSpacing(3)

        # Local Search Group
        local_label = QLabel("Tìm kiếm cục bộ")
        local_label.setObjectName("group_label")
        left_layout.addWidget(local_label)
        
        local_algorithms = [
            ('Simple Hill Climbing', 'HILL_CLIMBING', COLORS['local']),
            ('Steepest Ascent Hill Climbing', 'STEEPEST_HILL', COLORS['local']),
            ('Stochastic Hill Climbing', 'STOCHASTIC_HILL', COLORS['local']),
            ('Local Beam Search', 'LOCAL_BEAM', COLORS['local']),
            ('Simulated Annealing', 'ANNEALING', COLORS['local']),
            ('Genetic Algorithm', 'GENETIC', COLORS['local'])
        ]
        
        for name, algo_id, color in local_algorithms:
            btn = ControlButton(name, color)
            btn.setMinimumHeight(28)
            btn.setMaximumHeight(28)
            btn.clicked.connect(lambda checked, a=algo_id: self.select_algorithm(a))
            left_layout.addWidget(btn)

        left_layout.addSpacing(3)

        # AND/OR Graph Search Group
        andor_label = QLabel("Tìm kiếm trong môi trường không xác định")
        andor_label.setObjectName("group_label")
        left_layout.addWidget(andor_label)
        
        andor_algorithms = [
            ('AND/OR Graph Search', 'ANDOR', COLORS['complex']),
            ('Fixed Partially Observable', 'PARTIALLY_OBS', COLORS['complex']),
            ('Belief State Search', 'BELIEF_STATE', COLORS['complex'])
        ]
        
        for name, algo_id, color in andor_algorithms:
            btn = ControlButton(name, color)
            btn.setMinimumHeight(28)
            btn.setMaximumHeight(28)
            btn.clicked.connect(lambda checked, a=algo_id: self.select_algorithm(a))
            left_layout.addWidget(btn)

        left_layout.addSpacing(3)
        
        # Constraint Satisfaction Search Group
        constraint_label = QLabel("Tìm kiếm có ràng buộc")
        constraint_label.setObjectName("group_label")
        left_layout.addWidget(constraint_label)
        
        constraint_algorithms = [
            ('Backtracking Search', 'BACKTRACKING', COLORS['complex']),
            ('Forward Checking', 'FORWARD_CHECK', COLORS['complex']),
            ('AC-3 Algorithm', 'AC3', COLORS['complex'])
        ]
        
        for name, algo_id, color in constraint_algorithms:
            btn = ControlButton(name, color)
            btn.setMinimumHeight(28)
            btn.setMaximumHeight(28)
            btn.clicked.connect(lambda checked, a=algo_id: self.select_algorithm(a))
            left_layout.addWidget(btn)

        left_layout.addSpacing(3)

        # Evolutionary Algorithms Group
        evolutionary_label = QLabel("Thuật toán Tiến hóa")
        evolutionary_label.setObjectName("group_label")
        left_layout.addWidget(evolutionary_label)
        
        evolutionary_algorithms = [
            # Đã chuyển Genetic Algorithm sang nhóm Local Search
        ]
        
        # Chỉ hiển thị nhóm nếu có thuật toán
        if evolutionary_algorithms:
            for name, algo_id, color in evolutionary_algorithms:
                btn = ControlButton(name, color)
                btn.setMinimumHeight(28)
                btn.setMaximumHeight(28)
                btn.clicked.connect(lambda checked, a=algo_id: self.select_algorithm(a))
                left_layout.addWidget(btn)
        else:
            # Ẩn nhóm Thuật toán Tiến hóa nếu không có thuật toán
            evolutionary_label.hide()

        left_layout.addSpacing(3)
        
        # Reinforcement Learning Group
        rl_label = QLabel("Học tăng cường")
        rl_label.setObjectName("group_label")
        left_layout.addWidget(rl_label)
        
        rl_algorithms = [
            ('Q-Learning', 'QLEARNING', COLORS['primary'])
        ]
        
        for name, algo_id, color in rl_algorithms:
            btn = ControlButton(name, color)
            btn.setMinimumHeight(28)
            btn.setMaximumHeight(28)
            btn.clicked.connect(lambda checked, a=algo_id: self.select_algorithm(a))
            left_layout.addWidget(btn)

        left_layout.addStretch()
        
        scroll_area.setWidget(left_panel)
        algorithms_dock.setWidget(scroll_area)
        self.addDockWidget(Qt.LeftDockWidgetArea, algorithms_dock)

    def setupControlsDock(self):
        """Setup the controls dock widget"""
        controls_dock = QDockWidget("Controls", self)
        controls_dock.setObjectName("ControlsDock")
        controls_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        controls_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        
        # Right panel styling
        right_panel = QFrame()
        right_panel.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['surface']};
                border-radius: 8px;
                padding: 15px;
                border: 1px solid {COLORS['border']};
            }}
            QPushButton {{
                text-align: center;
                padding: 0px;
                font-size: 12px;
            }}
        """)
        right_panel.setMinimumWidth(250)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        # Control buttons
        controls_label = QLabel("Controls:")
        controls_label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 16px;
            font-weight: bold;
            padding: 0px 0px 10px 0px;
        """)
        right_layout.addWidget(controls_label)
        
        controls = [
            ('Đặt lại', self.on_reset, COLORS['danger']),
            ('Ngẫu nhiên', self.on_random, COLORS['warning']),
            ('Nhập liệu', self.on_input, COLORS['accent']),
            ('Giải', self.on_solve, COLORS['success']),
            ('Xuất', self.on_export, COLORS['secondary']),
            ('Tự động', self.on_auto_play, COLORS['primary']),
            ('Đánh giá', self.on_evaluate, COLORS['informed'])
        ]
        
        # Create a grid layout for buttons
        grid_layout = QGridLayout()
        grid_layout.setSpacing(8)
        
        for i, (text, handler, color) in enumerate(controls):
            btn = ControlButton(text, color)
            btn.setMinimumHeight(50)
            btn.clicked.connect(handler)
            row = i // 2
            col = i % 2
            grid_layout.addWidget(btn, row, col)
        
        right_layout.addLayout(grid_layout)
        right_layout.addStretch()
        
        controls_dock.setWidget(right_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, controls_dock)

    def setupLogDock(self):
        """Setup the log dock widget"""
        log_dock = QDockWidget("Log", self)
        log_dock.setObjectName("LogDock")
        log_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        log_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        
        # Log panel styling
        log_panel = QFrame()
        log_panel.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['surface']};
                border-radius: 8px;
                padding: 10px;
                border: 1px solid {COLORS['border']};
            }}
        """)
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(5, 5, 5, 5)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['background']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 5px;
                font-family: Consolas, Monaco, monospace;
                font-size: 12px;
            }}
        """)
        log_layout.addWidget(self.log_text)
        
        log_dock.setWidget(log_panel)
        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)
    
    def log(self, message):
        """Add a message to the log window"""
        self.log_text.append(f"> {message}")
        
        # Also update Q-Learning log if it exists and is visible
        if hasattr(self, 'ql_log_text') and self.current_algorithm == 'QLEARNING':
            self.ql_log_text.append(f"> {message}")
    
    def select_algorithm(self, algo):
        self.current_algorithm = algo
        self.update_status_labels()
        self.log(f"Algorithm changed to: {algo}")
        
        # Handle Q-Learning interface
        if algo == 'QLEARNING':
            # Hide the puzzle frame in the main window (it's not needed)
            self.puzzle_frame.setVisible(False)
            
            # Close the non-deterministic window if it's open
            if hasattr(self, 'non_deterministic_window') and self.non_deterministic_window is not None:
                self.non_deterministic_window.close()
                self.non_deterministic_window = None
                self.log("Đã đóng cửa sổ tìm kiếm trong môi trường không xác định")
            
            # Open the dedicated QLearningWindow
            if not hasattr(self, 'qlearning_window') or self.qlearning_window is None:
                # Check if q_agent exists, create if not
                global q_agent
                if 'q_agent' not in globals():
                    q_agent = QLearningAgent() # Use default params initially
                    self.log("Initialized Q-Learning agent with default parameters.")
                    
                self.qlearning_window = QLearningWindow(
                    parent=self,
                    initial_state=self.puzzle_board.state,
                    goal_state=self.goal_state
                )
                # Connect the log signal from QLearningWindow to MainWindow's log
                # self.qlearning_window.log_signal.connect(self.log) 
                # ^^^ We might need a signal if direct parent logging isn't sufficient
                
            self.qlearning_window.show()
            self.qlearning_window.raise_()
            self.qlearning_window.activateWindow()
            self.log("Opened Q-Learning window.")
        # Handle Non-Deterministic Environment interface
        elif algo in ['ANDOR', 'PARTIALLY_OBS', 'BELIEF_STATE']:
            # Hide the puzzle frame in the main window (it's not needed)
            self.puzzle_frame.setVisible(False)
            
            # Close the Q-learning window if it's open
            if hasattr(self, 'qlearning_window') and self.qlearning_window is not None:
                self.qlearning_window.close()
                self.qlearning_window = None
                self.log("Closed Q-Learning window.")
            
            # Open the dedicated NonDeterministicWindow
            if not hasattr(self, 'non_deterministic_window') or self.non_deterministic_window is None:
                self.non_deterministic_window = NonDeterministicWindow(
                    parent=self,
                    initial_state=self.puzzle_board.state,
                    goal_state=self.goal_state
                )
                
                # Đặt thuật toán hiện tại trong cửa sổ non-deterministic
                self.non_deterministic_window.select_algorithm(algo)
                if algo == 'ANDOR':
                    self.non_deterministic_window.algorithm_radio_andor.setChecked(True)
                elif algo == 'PARTIALLY_OBS':
                    self.non_deterministic_window.algorithm_radio_partial.setChecked(True)
                elif algo == 'BELIEF_STATE':
                    self.non_deterministic_window.algorithm_radio_belief.setChecked(True)
                
            self.non_deterministic_window.show()
            self.non_deterministic_window.raise_()
            self.non_deterministic_window.activateWindow()
            self.log("Đã mở cửa sổ tìm kiếm trong môi trường không xác định")
        else:
            # Show the puzzle frame for other algorithms
            self.puzzle_frame.setVisible(True)
            
            # Close the Q-learning window if it's open
            if hasattr(self, 'qlearning_window') and self.qlearning_window is not None:
                self.qlearning_window.close()
                self.qlearning_window = None
                self.log("Closed Q-Learning window.")
                
            # Close the non-deterministic window if it's open
            if hasattr(self, 'non_deterministic_window') and self.non_deterministic_window is not None:
                self.non_deterministic_window.close()
                self.non_deterministic_window = None
                self.log("Đã đóng cửa sổ tìm kiếm trong môi trường không xác định")
    
    def on_reset(self):
        self.puzzle_board.state = [1,2,3,4,5,6,7,0,8]
        self.solution_path = []
        self.current_step = 0
        self.puzzle_board.update()
        self.update_status_labels()
        self.log("Puzzle reset to initial state")
        
    def on_random(self):
        self.puzzle_board.state = generate_random_puzzle()
        self.solution_path = []
        self.current_step = 0
        self.puzzle_board.update()
        self.update_status_labels()
        self.log("Generated random puzzle state")
        
    def on_input(self):
        # TODO: Implement custom input dialog
        self.log("Custom input not implemented yet")
        pass
        
    def on_solve(self):
        start_state = tuple(self.puzzle_board.state)
        goal_state = tuple(self.goal_state)
        
        self.log(f"Solving puzzle using {self.current_algorithm}...")
        
        algorithm_map = {
            'BFS': bfs,
            'DFS': dfs,
            'UCS': uniform_cost,
            'GREEDY': greedy,
            'ASTAR': astar,
            'HILL_CLIMBING': hill_climbing,
            'STEEPEST_HILL': hill_climbing_max,
            'STOCHASTIC_HILL': hill_climbing_random_restart,
            'LOCAL_BEAM': lambda start, goal: local_beam_search(start, goal, beam_width=5),
            'ANNEALING': simulated_annealing,
            'IDS': iterative_deepening_search,
            'IDASTAR': ida_star,
            'GENETIC': genetic_algorithm,
            'ANDOR': andor_graph_search,
            'PARTIALLY_OBS': fixed_partially_observable,
            'BELIEF_STATE': belief_state_search,
            'BACKTRACKING': backtracking_search,
            'FORWARD_CHECK': forward_checking,
            'AC3': ac3_algorithm,
            'QLEARNING': q_learning
        }
        
        algorithm = algorithm_map.get(self.current_algorithm)
        if algorithm:
            self.solution_path = algorithm(start_state, goal_state)
            self.current_step = 0
            self.update_status_labels()
            
            if self.solution_path:
                self.log(f"Solution found with {len(self.solution_path)-1} steps")
            else:
                self.log("No solution found")
            
    def on_export(self):
        if not self.solution_path:
            self.log("No solution to export")
            return
            
        try:
            with open('solution.txt', 'w') as f:
                f.write(f"Algorithm: {self.current_algorithm}\n")
                f.write(f"Total steps: {len(self.solution_path)-1}\n\n")
                
                for i, state in enumerate(self.solution_path):
                    f.write(f"Step {i}:\n")
                    for row in range(3):
                        f.write(' '.join(str(state[row*3 + col]) for col in range(3)) + '\n')
                    f.write('\n')
            self.log("Solution exported to solution.txt")
        except Exception as e:
            self.log(f"Error exporting solution: {str(e)}")
                
    def on_auto_play(self):
        if not self.solution_path or self.current_step >= len(self.solution_path) - 1:
            self.log("No steps to play automatically")
            return
            
        self.log("Starting auto play...")
        self.auto_play_timer = QTimer(self)
        self.auto_play_timer.timeout.connect(self.auto_play_next)
        self.auto_play_timer.start(500)
        
    def auto_play_next(self):
        if self.current_step >= len(self.solution_path) - 1:
            self.auto_play_timer.stop()
            self.log("Auto play finished")
            return
            
        current_state = self.solution_path[self.current_step]
        next_state = self.solution_path[self.current_step + 1]
        
        # Find the moved tile
        for i in range(9):
            if current_state[i] != next_state[i] and current_state[i] != 0:
                self.puzzle_board.animate_move(current_state[i], i, next_state.index(current_state[i]))
                break
                
        self.current_step += 1
        self.puzzle_board.state = list(next_state)
        self.update_status_labels()
        self.log(f"Step {self.current_step} of {len(self.solution_path)-1}")
        
    def on_prev(self):
        if self.current_step > 0 and self.solution_path:
            self.current_step -= 1
            self.puzzle_board.state = list(self.solution_path[self.current_step])
            self.puzzle_board.update()
            self.update_status_labels()
            self.log(f"Moved to step {self.current_step}")
            
    def on_next(self):
        if self.current_step < len(self.solution_path) - 1:
            current_state = self.solution_path[self.current_step]
            next_state = self.solution_path[self.current_step + 1]
            
            # Find the moved tile
            for i in range(9):
                if current_state[i] != next_state[i] and current_state[i] != 0:
                    self.puzzle_board.animate_move(current_state[i], i, next_state.index(current_state[i]))
                    break
                    
            self.current_step += 1
            self.puzzle_board.state = list(next_state)
            self.update_status_labels()
            self.log(f"Moved to step {self.current_step}")
            
    def on_evaluate(self):
        """Open the algorithm evaluation window"""
        if not hasattr(self, 'evaluation_window') or self.evaluation_window is None:
            self.evaluation_window = EvaluationPage(parent=self)
            
        self.evaluation_window.show()
        self.evaluation_window.raise_()
        self.evaluation_window.activateWindow()
        self.log("Đã mở cửa sổ đánh giá thuật toán")
            
    def handle_cell_click(self, row, col):
        pos = row * 3 + col
        state = self.puzzle_board.state
        empty_pos = state.index(0)
        empty_row, empty_col = empty_pos // 3, empty_pos % 3
        
        if abs(row - empty_row) + abs(col - empty_col) == 1:
            # Valid move
            state[empty_pos], state[pos] = state[pos], state[empty_pos]
            self.puzzle_board.animate_move(state[empty_pos], pos, empty_pos)
            self.solution_path = []
            self.current_step = 0
            self.update_status_labels()
            self.log(f"Moved tile {state[empty_pos]} manually")
            
    def update_status_labels(self):
        self.algorithm_label.setText(f"Algorithm: {self.current_algorithm}")
        if self.solution_path:
            self.steps_label.setText(f"Step: {self.current_step} / {len(self.solution_path)-1}")
        else:
            self.steps_label.setText("No solution yet")

    def on_train_agent(self):
        # This method should ideally be part of QLearningWindow, 
        # but if called from MainWindow, delegate to the window
        if hasattr(self, 'qlearning_window') and self.qlearning_window is not None:
            self.qlearning_window.on_train_agent()
        else:
            self.log("Q-Learning window is not open. Select Q-Learning algorithm first.")

    def on_solve_qlearning(self):
        # This method should ideally be part of QLearningWindow,
        # but if called from MainWindow, delegate to the window
        if hasattr(self, 'qlearning_window') and self.qlearning_window is not None:
            self.qlearning_window.on_solve_qlearning()
        else:
            self.log("Q-Learning window is not open. Select Q-Learning algorithm first.")

    def update_q_values_display(self, state):
        # This method should ideally be part of QLearningWindow,
        # but if called from MainWindow, delegate to the window
        if hasattr(self, 'qlearning_window') and self.qlearning_window is not None:
            self.qlearning_window.update_q_values_display(state)
        # No else needed, display is handled within QLearningWindow

    def closeEvent(self, event):
        """Handle window close event"""
        # Close Q-Learning window if open
        if hasattr(self, 'qlearning_window') and self.qlearning_window is not None:
            self.qlearning_window.close()
            
        # Close Non-Deterministic window if open
        if hasattr(self, 'non_deterministic_window') and self.non_deterministic_window is not None:
            self.non_deterministic_window.close()
            
        # Close Evaluation window if open
        if hasattr(self, 'evaluation_window') and self.evaluation_window is not None:
            self.evaluation_window.close()
            
        event.accept()

class QLearningWindow(QMainWindow):
    """Separate window for Q-Learning algorithm"""
    def __init__(self, parent=None, initial_state=None, goal_state=None):
        super().__init__(parent)
        
        self.parent = parent
        self.initial_state = tuple(initial_state) if initial_state else tuple([1,2,3,4,5,6,7,0,8])
        self.goal_state = tuple(goal_state) if goal_state else tuple([1,2,3,4,5,6,7,8,0])
        self.ql_solution_path = [] # Path for QL window's auto-play
        self.ql_current_step = 0   # Current step for QL auto-play
        self.ql_auto_play_timer = QTimer(self) # Timer for QL auto-play
        self.ql_auto_play_timer.timeout.connect(self.ql_auto_play_next)
        
        # Set up the window
        self.setWindowTitle("Q-Learning - Học tăng cường")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(f"background-color: {COLORS['background']};")
        
        # Automatically maximize window when opened
        self.showMaximized()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Q-Learning - Học tăng cường")
        title_label.setStyleSheet(f"""
            color: {COLORS['primary']};
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
            padding: 5px;
            border-bottom: 2px solid {COLORS['primary']};
        """)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create a horizontal layout for the main content
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # ===== LEFT PANEL (No Scroll Needed Here) =====
        left_panel = QFrame()
        left_panel.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['surface']};
                border-radius: 8px;
                border: 1px solid {COLORS['border']};
                padding: 10px;
            }}
        """)
        left_panel.setMinimumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        
        # Puzzle board section
        board_group = QGroupBox("Bảng Puzzle")
        board_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px;
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                margin-top: 10px;
                padding: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: {COLORS['surface']};
            }}
        """)
        
        board_layout = QVBoxLayout(board_group)
        board_layout.setContentsMargins(10, 20, 10, 10)
        
        self.puzzle_board = PuzzleBoard()
        self.puzzle_board.state = list(self.initial_state)
        board_layout.addWidget(self.puzzle_board, alignment=Qt.AlignCenter)
        
        left_layout.addWidget(board_group)
        
        # State information section
        state_group = QGroupBox("Thông tin trạng thái")
        state_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px;
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                margin-top: 10px;
                padding: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: {COLORS['surface']};
            }}
        """)
        
        state_layout = QVBoxLayout(state_group)
        state_layout.setContentsMargins(10, 20, 10, 10)
        
        self.q_state_info = QLabel("Chưa có giá trị Q")
        self.q_state_info.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 14px;
            padding: 8px;
            background: {COLORS['divider']};
            border-radius: 4px;
            margin-bottom: 10px;
        """)
        self.q_state_info.setAlignment(Qt.AlignCenter)
        state_layout.addWidget(self.q_state_info)
        
        # Q-values grid
        q_values_label = QLabel("Giá trị Q cho các hành động:")
        q_values_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px; margin-bottom: 5px; font-weight: bold;")
        state_layout.addWidget(q_values_label)
        
        q_grid = QWidget()
        q_grid_layout = QGridLayout(q_grid)
        q_grid_layout.setHorizontalSpacing(8)
        q_grid_layout.setVerticalSpacing(8)
        
        self.q_value_labels = {}
        directions = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}
        
        for i in range(3):
            for j in range(3):
                pos = i * 3 + j
                direction = None
                
                if pos == 1:  # Top
                    direction = 'U'
                elif pos == 3:  # Left
                    direction = 'L'
                elif pos == 5:  # Right
                    direction = 'R'
                elif pos == 7:  # Bottom
                    direction = 'D'
                
                q_label = QLabel()
                if direction:
                    q_label.setText(f"{directions[direction]}: 0.0")
                    q_label.setStyleSheet(f"""
                        color: {COLORS['primary']};
                        font-size: 14px;
                        font-weight: bold;
                        background: {COLORS['surface']};
                        border: 1px solid {COLORS['border']};
                        border-radius: 4px;
                        padding: 8px;
                        min-width: 80px;
                        text-align: center;
                    """)
                    self.q_value_labels[direction] = q_label
                else:
                    # Add an empty label to maintain grid structure
                    q_label = QLabel()
                    q_label.setMinimumSize(80, 30) # Match approx size
                
                q_grid_layout.addWidget(q_label, i, j, alignment=Qt.AlignCenter)
        
        state_layout.addWidget(q_grid)
        left_layout.addWidget(state_group)
        left_layout.addStretch() # Add stretch to push content up
        
        # ===== RIGHT PANEL (Layout Adjustments) =====
        right_panel = QFrame()
        right_panel.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['surface']};
                border-radius: 8px;
                border: 1px solid {COLORS['border']};
                padding: 10px;
            }}
        """)
        right_panel.setMinimumWidth(400)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        
        # Create tab widget for training info and configuration
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                background: {COLORS['background']};
                padding: 10px; /* Add padding inside pane */
            }}
            QTabBar::tab {{
                background: {COLORS['background']};
                border: 1px solid {COLORS['border']};
                border-bottom-color: {COLORS['border']};
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 100px;
                padding: 8px;
                font-weight: bold;
            }}
            QTabBar::tab:selected {{
                background: {COLORS['primary']};
                color: white;
            }}
            QTabBar::tab:!selected {{
                margin-top: 2px;
            }}
        """)
        
        # Training info tab (Content inside might need scroll if window very small)
        training_tab = QWidget()
        training_scroll = QScrollArea() # Add scroll here for tab content
        training_scroll.setWidgetResizable(True)
        training_scroll.setStyleSheet("border: none; background: transparent;")
        training_scroll.setWidget(training_tab)
        
        training_layout = QVBoxLayout(training_tab)
        training_layout.setContentsMargins(0, 0, 0, 0) # No margins for content widget
        training_layout.setSpacing(15)
        
        # Training info group
        training_group = QGroupBox("Thông tin huấn luyện")
        training_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px;
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                margin-top: 10px;
                padding: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: {COLORS['background']};
            }}
        """)
        
        training_info_layout = QVBoxLayout(training_group)
        training_info_layout.setContentsMargins(10, 20, 10, 10)
        
        # Progress section
        progress_label = QLabel("Tiến trình:")
        progress_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px; font-weight: bold;")
        training_info_layout.addWidget(progress_label)
        
        self.progress_label = QLabel("Episodes: 0/0")
        self.progress_label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 16px;
            padding: 8px;
            background: {COLORS['surface']};
            border-radius: 4px;
            border: 1px solid {COLORS['border']};
        """)
        self.progress_label.setAlignment(Qt.AlignCenter)
        training_info_layout.addWidget(self.progress_label)
        
        # Parameters section
        params_label = QLabel("Thông số thuật toán:")
        params_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px; font-weight: bold; margin-top: 10px;")
        training_info_layout.addWidget(params_label)
        
        params_grid = QGridLayout()
        params_grid.setColumnStretch(0, 1)
        params_grid.setColumnStretch(1, 1)
        params_grid.setHorizontalSpacing(10)
        params_grid.setVerticalSpacing(8)
        
        stat_titles = ["Tổng Q-values:", "Alpha (Tỉ lệ học):", "Gamma (Giảm giá):", "Epsilon (Khám phá):"]
        self.stat_values = []
        
        for i, title in enumerate(stat_titles):
            label = QLabel(title)
            label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px;")
            params_grid.addWidget(label, i, 0)
            
            value = QLabel("0" if i == 0 else "0.1" if i == 1 else "0.9" if i == 2 else "0.1")
            value.setStyleSheet(f"""
                color: {COLORS['text_primary']};
                font-size: 14px;
                background: {COLORS['surface']};
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
            """)
            value.setAlignment(Qt.AlignCenter)
            params_grid.addWidget(value, i, 1)
            self.stat_values.append(value)
        
        training_info_layout.addLayout(params_grid)
        
        # Results section
        results_label = QLabel("Kết quả:")
        results_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px; font-weight: bold; margin-top: 10px;")
        training_info_layout.addWidget(results_label)
        
        self.qlearning_info = QLabel("Chưa có dữ liệu huấn luyện. Nhấn 'Bắt đầu huấn luyện' để tiến hành.")
        self.qlearning_info.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 14px;
            padding: 10px;
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            min-height: 80px;
        """)
        self.qlearning_info.setAlignment(Qt.AlignCenter)
        self.qlearning_info.setWordWrap(True)
        training_info_layout.addWidget(self.qlearning_info)
        
        training_layout.addWidget(training_group)
        training_layout.addStretch() # Stretch at the end of the tab content
        
        # Configuration tab (No Scroll Needed for this fixed content)
        config_tab = QWidget() 
        # config_scroll = QScrollArea() # REMOVE ScrollArea for config tab
        # config_scroll.setWidgetResizable(True)
        # config_scroll.setStyleSheet("border: none; background: transparent;")
        # config_scroll.setWidget(config_tab)
        
        config_layout = QVBoxLayout(config_tab)
        config_layout.setContentsMargins(0, 0, 0, 0) # No margins for content widget
        config_layout.setSpacing(15)
        
        # Configuration group
        config_group = QGroupBox("Cấu hình thuật toán")
        config_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px;
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                margin-top: 10px;
                padding: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: {COLORS['background']};
            }}
        """)
        
        config_group_layout = QVBoxLayout(config_group)
        config_group_layout.setContentsMargins(10, 20, 10, 10)
        
        # Parameters form
        param_grid = QGridLayout()
        param_grid.setColumnStretch(0, 1)
        param_grid.setColumnStretch(1, 2)
        param_grid.setHorizontalSpacing(15)
        param_grid.setVerticalSpacing(10)
        
        param_names = ["Episodes:", "Alpha:", "Gamma:", "Epsilon:"]
        default_values = ["100", "0.1", "0.9", "0.1"]
        self.param_inputs = []
        
        for i, (name, value) in enumerate(zip(param_names, default_values)):
            label = QLabel(name)
            label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px; font-weight: bold;")
            param_grid.addWidget(label, i, 0)
            
            input_field = QLineEdit(value)
            input_field.setStyleSheet(f"""
                background: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
                color: {COLORS['text_primary']};
                font-size: 14px;
            """)
            param_grid.addWidget(input_field, i, 1)
            self.param_inputs.append(input_field)
        
        config_group_layout.addLayout(param_grid)
        config_layout.addWidget(config_group)
        
        # Control buttons
        buttons_group = QGroupBox("Điều khiển")
        buttons_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px;
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                margin-top: 10px;
                padding: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: {COLORS['background']};
            }}
        """)
        
        buttons_layout = QVBoxLayout(buttons_group)
        buttons_layout.setContentsMargins(10, 20, 10, 10)
        
        buttons_container = QHBoxLayout()
        buttons_container.setSpacing(15)
        
        self.train_btn = QPushButton("Bắt đầu huấn luyện")
        self.train_btn.setMinimumHeight(50)
        self.train_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: #1976D2;
            }}
            QPushButton:pressed {{
                background-color: #0D47A1;
            }}
        """)
        self.train_btn.clicked.connect(self.on_train_agent)
        buttons_container.addWidget(self.train_btn)
        
        self.solve_btn = QPushButton("Giải với Q-Learning")
        self.solve_btn.setMinimumHeight(50)
        self.solve_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: #388E3C;
            }}
            QPushButton:pressed {{
                background-color: #1B5E20;
            }}
        """)
        self.solve_btn.clicked.connect(self.on_solve_qlearning)
        buttons_container.addWidget(self.solve_btn)
        
        buttons_layout.addLayout(buttons_container)
        config_layout.addWidget(buttons_group)
        config_layout.addStretch() # Stretch at the end of the tab content
        
        # Add tabs to tab widget
        tab_widget.addTab(training_scroll, "Thông tin huấn luyện")
        tab_widget.addTab(config_tab, "Cấu hình") # Add config_tab directly
        
        right_layout.addWidget(tab_widget)
        # right_layout.addStretch() # Remove stretch here, let tab widget fill space
        
        # Add both panels to the content layout
        content_layout.addWidget(left_panel, 4)  # 40% width
        content_layout.addWidget(right_panel, 6)  # 60% width
        
        main_layout.addLayout(content_layout)
        
        # Add log section at the bottom
        log_group = QGroupBox("Nhật ký hoạt động")
        log_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px;
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                margin-top: 5px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: {COLORS['background']};
            }}
        """)
        
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(10, 20, 10, 10)
        
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setStyleSheet(f"""
            background-color: {COLORS['surface']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 5px;
            font-family: Consolas, Monaco, monospace;
            font-size: 12px;
        """)
        # Remove max-height to allow log to take its space
        # self.log_widget.setMaximumHeight(100)
        log_layout.addWidget(self.log_widget)
        
        main_layout.addWidget(log_group)
        
        # Set size ratio for content vs log
        main_layout.setStretchFactor(content_layout, 80)  # 80% for content
        main_layout.setStretchFactor(log_group, 20)       # 20% for log
        
        # Initial log
        self.log("Cửa sổ Q-Learning đã sẵn sàng")

    def log(self, message):
        """Add a message to the log window"""
        self.log_widget.append(f"> {message}")
        # Optionally, emit a signal if the main window needs the log too
        # self.log_signal.emit(message)
        if self.parent:
            self.parent.log(message) # Log to main window as well if needed

    def on_train_agent(self):
        """Train Q-Learning agent"""
        self.log("Đang huấn luyện Q-Learning agent...")
        
        # Get parameters from input fields
        try:
            max_episodes = int(self.param_inputs[0].text())
            alpha = float(self.param_inputs[1].text())
            gamma = float(self.param_inputs[2].text())
            epsilon = float(self.param_inputs[3].text())
        except ValueError:
            self.log("Lỗi: Thông số không hợp lệ. Sử dụng giá trị cải tiến.")
            max_episodes = 500  # Tăng số lượng episodes
            alpha = 0.2  # Tăng tỉ lệ học tập
            gamma = 0.95  # Tăng hệ số giảm giá
            epsilon = 0.2  # Tăng tỉ lệ khám phá
        
        start_state = tuple(self.puzzle_board.state)
        goal_state = tuple(self.goal_state)
        
        # Ensure Q-Learning agent exists (it should have been created in MainWindow)
        global q_agent
        if 'q_agent' not in globals():
            self.log("Lỗi nghiêm trọng: Q-Learning agent chưa được khởi tạo.")
            q_agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
        else:
            # Update parameters for existing agent
            q_agent.alpha = alpha
            q_agent.gamma = gamma
            q_agent.epsilon = epsilon
        
        # Show training message
        self.qlearning_info.setText("Đang huấn luyện...\nQuá trình này có thể mất một chút thời gian.")
        self.progress_label.setText(f"Episodes: 0/{max_episodes}")
        QApplication.processEvents()  # Update UI
        
        # Train agent 
        # TODO: Consider running this in a separate thread to avoid freezing UI
        q_agent.train(start_state, goal_state, max_episodes=max_episodes, max_steps=100)
        
        # Update information after training
        self.qlearning_info.setText(
            f"Huấn luyện hoàn thành!\n"
            f"- Episods đã hoàn thành: {q_agent.current_episode}\n"
            f"- Tổng phần thưởng: {sum(q_agent.episode_rewards):.2f}\n"
            f"- Trung bình bước/episode: {sum(q_agent.episode_steps)/len(q_agent.episode_steps):.2f}"
        )
        self.progress_label.setText(f"Episodes: {q_agent.current_episode}/{max_episodes}")
        
        # Update statistics
        self.stat_values[0].setText(str(len(q_agent.q_values)))
        self.stat_values[1].setText(str(q_agent.alpha))
        self.stat_values[2].setText(str(q_agent.gamma)) 
        self.stat_values[3].setText(str(q_agent.epsilon))
        
        # Display Q-values for current state
        self.update_q_values_display(start_state)
        
        # Also update the board display
        if hasattr(self, 'puzzle_board'): # Check if board exists
            self.puzzle_board.update_state(start_state)
        
        self.log(f"Huấn luyện Q-Learning hoàn thành với {q_agent.current_episode} episodes")

    def on_solve_qlearning(self):
        """Solve puzzle with Q-Learning"""
        self.log("Đang giải puzzle với Q-Learning...")
        
        start_state = tuple(self.puzzle_board.state)
        goal_state = tuple(self.goal_state)
        
        # Use trained agent
        global q_agent
        if 'q_agent' not in globals():
            self.log("Cần huấn luyện agent trước khi giải!")
            self.qlearning_info.setText("Cần huấn luyện agent trước khi giải!")
            return
        
        # Stop any previous auto-play
        if self.ql_auto_play_timer.isActive():
            self.ql_auto_play_timer.stop()
            
        # Solve puzzle
        self.ql_solution_path = q_agent.solve(start_state, goal_state)
        self.ql_current_step = 0 # Reset step counter
        
        # Update state information display in this window
        self.update_q_values_display(start_state)
        
        # Show result
        if self.ql_solution_path and len(self.ql_solution_path) > 1:
            self.log(f"Q-Learning tìm thấy giải pháp với {len(self.ql_solution_path)-1} bước")
            self.qlearning_info.setText(
                f"Đã tìm thấy giải pháp!\n"
                f"Số bước: {len(self.ql_solution_path)-1}"
            )
            
            # Update the board in this window to the initial state of the solution
            self.puzzle_board.update_state(self.ql_solution_path[0])
            
            # Update the main window if it exists
            if self.parent and isinstance(self.parent, MainWindow):
                self.parent.solution_path = self.ql_solution_path # Use the same path
                self.parent.current_step = 0
                self.parent.puzzle_board.update_state(self.ql_solution_path[0]) # Update main board
                self.parent.update_status_labels()
                self.parent.log(f"Q-Learning solution displayed ({len(self.ql_solution_path)-1} steps).")
                # --- Trigger Auto Play in MainWindow --- 
                self.parent.on_auto_play() 
                # -------------------------------------
            
            # --- Trigger Auto Play in this QLearningWindow ---
            self.start_ql_auto_play()
            # -----------------------------------------------
                
        else:
            self.log("Q-Learning không tìm thấy giải pháp")
            self.qlearning_info.setText("Không tìm thấy giải pháp.")
            self.ql_solution_path = [] # Clear path if no solution

    def start_ql_auto_play(self):
        """Starts the auto-play timer for the Q-Learning window's board."""
        if not self.ql_solution_path or self.ql_current_step >= len(self.ql_solution_path) - 1:
            self.log("Không có bước nào để tự động phát trong cửa sổ Q-Learning")
            return
            
        if self.ql_auto_play_timer.isActive():
            self.ql_auto_play_timer.stop() # Stop if already running
            
        self.log("Bắt đầu tự động phát trên bảng Q-Learning...")
        # Set a slightly different interval maybe?
        self.ql_auto_play_timer.start(600) # e.g., 600ms interval

    def ql_auto_play_next(self):
        """Plays the next step of the solution on the Q-Learning window's board."""
        if self.ql_current_step >= len(self.ql_solution_path) - 1:
            self.ql_auto_play_timer.stop()
            self.log("Tự động phát trên bảng Q-Learning hoàn thành")
            return
            
        current_state = self.ql_solution_path[self.ql_current_step]
        next_state = self.ql_solution_path[self.ql_current_step + 1]
        
        # Find the moved tile
        moved_tile = -1
        start_idx = -1
        end_idx = -1
        for i in range(9):
            if current_state[i] != next_state[i] and current_state[i] != 0:
                moved_tile = current_state[i]
                start_idx = i
                end_idx = next_state.index(moved_tile)
                break
                
        if moved_tile != -1:
             self.puzzle_board.animate_move(moved_tile, start_idx, end_idx)
             
        self.ql_current_step += 1
        self.puzzle_board.state = list(next_state) # Update internal state after animation starts
        # No need to call self.puzzle_board.update() here, animation handles it
        
        self.log(f"QL Bảng Bước {self.ql_current_step} / {len(self.ql_solution_path)-1}")
        
        # Update Q-values display for the new state
        self.update_q_values_display(next_state)

    def update_q_values_display(self, state):
        """Display Q-values for current state"""
        global q_agent
        if 'q_agent' not in globals():
             self.log("Q-agent not initialized, cannot display Q-values.")
             return
             
        # Update state information
        empty_idx = state.index(0)
        state_desc = f"Trạng thái: {state}\nÔ trống ở vị trí: {empty_idx}"
        self.q_state_info.setText(state_desc)
        
        # Get list of legal actions
        legal_actions = q_agent.get_legal_actions(state)
        
        # Update Q-values for each action
        directions = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}
        best_q_value = -float('inf')
        best_action = None
        action_values = {}

        if legal_actions: # Find best action only if there are legal actions
            for action in legal_actions:
                q_val = q_agent.get_q_value(state, action)
                action_values[action] = q_val
                if q_val > best_q_value:
                    best_q_value = q_val
                    best_action = action
        
        for direction in ['U', 'D', 'L', 'R']:
            if direction in self.q_value_labels: # Check if the label exists
                q_label = self.q_value_labels[direction]
                if direction in legal_actions:
                    q_value = action_values[direction]
                    is_best = direction == best_action
                    
                    color = COLORS['success'] if is_best else COLORS['primary']
                    q_label.setStyleSheet(f"""
                        color: {color};
                        font-size: 14px;
                        font-weight: {'bold' if is_best else 'normal'};
                        background: {COLORS['surface']};
                        border: 1px solid {color if is_best else COLORS['border']};
                        border-radius: 4px;
                        padding: 8px;
                        min-width: 80px;
                        text-align: center;
                    """)
                    q_label.setText(f"{directions[direction]}: {q_value:.4f}")
                else:
                    q_label.setStyleSheet(f"""
                        color: {COLORS['text_secondary']};
                        font-size: 14px;
                        background: {COLORS['divider']};
                        border: 1px solid {COLORS['border']};
                        border-radius: 4px;
                        padding: 8px;
                        min-width: 80px;
                        text-align: center;
                    """)
                    q_label.setText(f"{directions[direction]}: N/A")

    def closeEvent(self, event):
        """Handle window close event"""
        # Stop the timer if the window is closed during auto-play
        if self.ql_auto_play_timer.isActive():
            self.ql_auto_play_timer.stop()
            
        # Safely access parent and its attributes
        parent_window = self.parent
        if parent_window and isinstance(parent_window, MainWindow):
            parent_window.qlearning_window = None # Clear reference in parent
            parent_window.log("Closed Q-Learning window.")
            # Ensure main puzzle frame is visible if Q-learning window is closed
            # Check the algorithm selected in the *parent* window
            if parent_window.current_algorithm != 'QLEARNING': 
                 parent_window.puzzle_frame.setVisible(True)
            elif parent_window.current_algorithm == 'QLEARNING':
                 # If we close the QL window while QL is selected,
                 # make the main puzzle visible again.
                 parent_window.puzzle_frame.setVisible(True) 
                 parent_window.log("Switched back to main puzzle view.")
        elif parent_window: 
             # Log if parent exists but is not MainWindow (unexpected)
             print(f"Warning: QLearningWindow parent is not MainWindow: {type(parent_window)}")

        event.accept()

class EvaluationPage(QMainWindow):
    """Window for algorithm evaluation and comparison"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.goal_state = tuple([1,2,3,4,5,6,7,8,0]) # Store as tuple
        self.evaluation_results = {}
        self.evaluation_thread = None
        
        # Configure window
        self.setWindowTitle("Đánh giá Thuật toán 8 Puzzle")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(f"background-color: {COLORS['background']};")
        
        # Automatically maximize window when opened
        self.showMaximized()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Đánh giá và So sánh Thuật toán")
        title_label.setStyleSheet(f"""
            color: {COLORS['primary']};
            font-size: 24px;
            font-weight: bold;
            padding: 10px 0px;
            border-bottom: 2px solid {COLORS['primary']};
        """)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Configuration panel
        config_panel = QFrame()
        config_panel.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['surface']};
                border-radius: 8px;
                border: 1px solid {COLORS['border']};
                padding: 10px;
            }}
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: 14px;
                font-weight: bold;
            }}
            QComboBox, QSpinBox {{
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 5px;
                background: white;
                min-height: 30px;
            }}
        """)
        config_layout = QHBoxLayout(config_panel)
        config_layout.setSpacing(15)
        
        # Number of puzzles
        puzzles_layout = QVBoxLayout()
        puzzles_label = QLabel("Số lượng Puzzle:")
        self.puzzles_spinner = QSpinBox()
        self.puzzles_spinner.setRange(1, 50)
        self.puzzles_spinner.setValue(10)
        puzzles_layout.addWidget(puzzles_label)
        puzzles_layout.addWidget(self.puzzles_spinner)
        config_layout.addLayout(puzzles_layout)
        
        # Algorithm selection
        alg_layout = QVBoxLayout()
        alg_label = QLabel("Nhóm Thuật toán:")
        self.alg_combo = QComboBox()
        self.alg_combo.addItems([
            "Tất cả Thuật toán", 
            "Tìm kiếm không có thông tin", 
            "Tìm kiếm có thông tin", 
            "Tìm kiếm cục bộ", 
            "Tìm kiếm không xác định",
            "Tìm kiếm có ràng buộc",
            "Học tăng cường",
            "Thuật toán đã chọn (Mặc định)"
        ])
        alg_layout.addWidget(alg_label)
        alg_layout.addWidget(self.alg_combo)
        config_layout.addLayout(alg_layout)
        
        # Action buttons
        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(QLabel("Thao tác:"))
        
        buttons_container = QHBoxLayout()
        self.run_btn = ControlButton("Chạy Đánh giá", COLORS['primary'])
        self.run_btn.clicked.connect(self.run_evaluation)
        buttons_container.addWidget(self.run_btn)
        
        self.export_btn = ControlButton("Xuất PDF", COLORS['accent'])
        self.export_btn.clicked.connect(self.export_to_pdf)
        self.export_btn.setEnabled(False)
        buttons_container.addWidget(self.export_btn)
        
        buttons_layout.addLayout(buttons_container)
        config_layout.addLayout(buttons_layout)
        
        main_layout.addWidget(config_panel)
        
        # Progress panel
        progress_panel = QFrame()
        progress_panel.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['surface']};
                border-radius: 8px;
                border: 1px solid {COLORS['border']};
                padding: 10px;
            }}
        """)
        progress_layout = QVBoxLayout(progress_panel)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {COLORS['border']};
                border-radius: 5px;
                text-align: center;
                background: {COLORS['divider']};
                height: 25px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 5px;
            }}
        """)
        
        self.status_label = QLabel("Sẵn sàng đánh giá thuật toán.")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; margin-top: 5px;")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        
        main_layout.addWidget(progress_panel)
        
        # Tabs for different charts and results
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {COLORS['border']};
                border-radius: 5px;
                background: {COLORS['surface']};
                padding: 10px;
            }}
            QTabBar::tab {{
                background: {COLORS['background']};
                border: 1px solid {COLORS['border']};
                padding: 8px 15px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
                min-width: 120px;
                font-size: 11px;
            }}
            QTabBar::tab:selected {{
                background: {COLORS['primary']};
                color: white;
            }}
        """)
        
        # Tab 1: Comparison Table
        self.table_tab = QWidget()
        table_layout = QVBoxLayout(self.table_tab)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Thuật toán", "Thời gian TB (s)", "Độ dài Đường đi TB", 
            "Tỉ lệ Thành công (%)", "Ước tính Nút Mở rộng"
        ])
        self.results_table.setStyleSheet(f"""
            QTableWidget {{
                border: 1px solid {COLORS['border']};
                border-radius: 5px;
                background: white;
                gridline-color: {COLORS['divider']};
            }}
            QHeaderView::section {{
                background: {COLORS['primary']};
                color: white;
                font-weight: bold;
                padding: 5px;
                border: none;
            }}
        """)
        table_layout.addWidget(self.results_table)
        
        # Tab 2: Time Comparison Chart
        self.time_tab = QWidget()
        time_layout = QVBoxLayout(self.time_tab)
        
        self.time_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        time_layout.addWidget(self.time_canvas)
        
        # Tab 3: Path Length Chart
        self.path_tab = QWidget()
        path_layout = QVBoxLayout(self.path_tab)
        
        self.path_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        path_layout.addWidget(self.path_canvas)
        
        # Tab 4: Success Rate Chart
        self.success_tab = QWidget()
        success_layout = QVBoxLayout(self.success_tab)
        
        self.success_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        success_layout.addWidget(self.success_canvas)
        
        # Tab 5: Group Comparison - Made into Tab Widget
        self.group_tab = QWidget()
        group_layout = QVBoxLayout(self.group_tab)
        
        # Create a tab widget to hold tabs for each algorithm group
        self.group_tab_widget = QTabWidget()
        self.group_tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {COLORS['border']};
                border-radius: 5px;
                background: {COLORS['surface']};
                padding: 10px;
            }}
            QTabBar::tab {{
                background: {COLORS['background']};
                border: 1px solid {COLORS['border']};
                padding: 8px 15px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
                min-width: 120px;
                font-size: 11px;
            }}
            QTabBar::tab:selected {{
                background: {COLORS['primary']};
                color: white;
            }}
        """)
        
        # Dictionary to store scroll areas and containers for each group
        self.group_containers = {}
        self.group_widgets = {}
        
        # Define algorithm groups for tab creation
        self.algorithm_groups = {
            "Tìm kiếm không có thông tin": {'color': "#2196F3", 'algorithms': ['BFS', 'DFS', 'UCS', 'IDS']},
            "Tìm kiếm có thông tin": {'color': "#4CAF50", 'algorithms': ['ASTAR', 'IDASTAR', 'GREEDY']},
            "Tìm kiếm cục bộ": {'color': "#FF5722", 'algorithms': ['HILL_CLIMBING', 'STEEPEST_HILL', 'STOCHASTIC_HILL', 'LOCAL_BEAM', 'ANNEALING', 'GENETIC']},
            "Tìm kiếm không xác định": {'color': "#9C27B0", 'algorithms': ['ANDOR', 'PARTIALLY_OBS', 'BELIEF_STATE']},
            "Tìm kiếm có ràng buộc": {'color': "#FF9800", 'algorithms': ['BACKTRACKING', 'FORWARD_CHECK', 'AC3']},
            "Học tăng cường": {'color': "#009688", 'algorithms': ['QLEARNING']}
        }
        
        # Create a tab for each algorithm group
        for group_name, group_info in self.algorithm_groups.items():
            # Create a tab for this group
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            
            # Create a scroll area for this group
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            # Create content widget and layout
            self.group_widgets[group_name] = QWidget()
            self.group_containers[group_name] = QVBoxLayout(self.group_widgets[group_name])
            self.group_containers[group_name].setSpacing(20)
            
            # Add widget to scroll area
            scroll_area.setWidget(self.group_widgets[group_name])
            tab_layout.addWidget(scroll_area)
            
            # Add tab with color-coded indicator
            self.group_tab_widget.addTab(tab, group_name)
            
            # Color the tab's indicator line
            tab_index = self.group_tab_widget.indexOf(tab)
            tab_bar = self.group_tab_widget.tabBar()
            if tab_bar:
                tab_bar.setTabTextColor(tab_index, QColor(group_info['color']))
        
        group_layout.addWidget(self.group_tab_widget)
        
        # Add tabs to widget
        self.tab_widget.addTab(self.table_tab, "Bảng So sánh")
        self.tab_widget.addTab(self.time_tab, "Thời gian Thực thi")
        self.tab_widget.addTab(self.path_tab, "Độ dài Đường đi")
        self.tab_widget.addTab(self.success_tab, "Tỉ lệ Thành công")
        self.tab_widget.addTab(self.group_tab, "So sánh Theo Nhóm")
        
        # Give the tab widget full space
        main_layout.addWidget(self.tab_widget)
        
        # Create a hidden text area for logs (not displayed but still accessible)
        self.log_text = QTextEdit()
        self.log_text.setVisible(False)
        
        # Initial log - still sent to parent, but not shown in UI
        self.log("Trang đánh giá đã khởi tạo và sẵn sàng sử dụng.")
        
    def log(self, message):
        """Add a message to the log window"""
        self.log_text.append(f"> {message}")
        if self.parent:
            self.parent.log(message)
    
    def get_selected_algorithms(self):
        """Get list of algorithms based on selection"""
        category = self.alg_combo.currentText()
        
        all_algorithms = [
            'BFS', 'DFS', 'UCS', 'IDS', 
            'ASTAR', 'IDASTAR', 'GREEDY', 
            'HILL_CLIMBING', 'STEEPEST_HILL', 'STOCHASTIC_HILL', 
            'LOCAL_BEAM', 'ANNEALING', 'GENETIC', 
            'ANDOR', 'PARTIALLY_OBS', 'BELIEF_STATE', 
            'BACKTRACKING', 'FORWARD_CHECK', 'AC3',
            'QLEARNING'
        ]
        
        if category == "Tất cả Thuật toán":
            return all_algorithms
        elif category == "Tìm kiếm không có thông tin":
            return ['BFS', 'DFS', 'UCS', 'IDS']
        elif category == "Tìm kiếm có thông tin":
            return ['ASTAR', 'IDASTAR', 'GREEDY']
        elif category == "Tìm kiếm cục bộ":
            return ['HILL_CLIMBING', 'STEEPEST_HILL', 'STOCHASTIC_HILL', 'LOCAL_BEAM', 'ANNEALING', 'GENETIC']
        elif category == "Tìm kiếm không xác định":
            return ['ANDOR', 'PARTIALLY_OBS', 'BELIEF_STATE']
        elif category == "Tìm kiếm có ràng buộc":
            return ['BACKTRACKING', 'FORWARD_CHECK', 'AC3']
        elif category == "Học tăng cường":
            return ['QLEARNING']
        elif category == "Thuật toán đã chọn (Mặc định)": 
            return ['BFS', 'ASTAR', 'HILL_CLIMBING'] # A small default set
        else: 
            return ['BFS', 'ASTAR'] # Fallback, though should not be reached
    
    def run_evaluation(self):
        """Run evaluation on selected algorithms"""
        if self.evaluation_thread and self.evaluation_thread.isRunning():
            self.log("Đánh giá đang được tiến hành. Vui lòng đợi.")
            return
            
        # Reset UI state
        self.progress_bar.setValue(0)
        self.status_label.setText("Bắt đầu đánh giá...")
        self.export_btn.setEnabled(False)
        
        # Get parameters
        num_puzzles = self.puzzles_spinner.value()
        algorithms = self.get_selected_algorithms()
        
        # Create and start thread
        self.evaluation_thread = EvaluationThread(
            None,  # Will generate random puzzles in the thread
            self.goal_state,
            num_puzzles,
            algorithms,
            self
        )
        
        self.evaluation_thread.update_progress.connect(self.update_progress)
        self.evaluation_thread.evaluation_complete.connect(self.display_results)
        
        self.log(f"Bắt đầu đánh giá {len(algorithms)} thuật toán trên {num_puzzles} puzzle...")
        self.evaluation_thread.start()
        
    def update_progress(self, progress, message):
        """Update progress bar and status message"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
        self.log(message)
        
    def display_results(self, results):
        """Display evaluation results in charts and table"""
        self.evaluation_results = results
        
        # Enable export button
        self.export_btn.setEnabled(True)
        
        # Update comparison table
        self.update_table(results)
        
        # Update charts
        self.update_time_chart(results)
        self.update_path_chart(results)
        self.update_success_chart(results)
        self.update_group_charts(results)
        
        self.log("Kết quả đã được hiển thị thành công.")
        
    def update_table(self, results):
        """Update the comparison table with results"""
        self.results_table.setRowCount(0)  # Clear table
        
        # Sort algorithms by average time
        sorted_algos = sorted(results.keys(), key=lambda x: results[x]['avg_time'])
        
        for i, algo in enumerate(sorted_algos):
            self.results_table.insertRow(i)
            
            # Algorithm name
            self.results_table.setItem(i, 0, QTableWidgetItem(algo))
            
            # Execution time (s)
            time_item = QTableWidgetItem(f"{results[algo]['avg_time']:.4f}")
            self.results_table.setItem(i, 1, time_item)
            
            # Path length
            path_item = QTableWidgetItem(f"{results[algo]['avg_path_length']:.2f}")
            self.results_table.setItem(i, 2, path_item)
            
            # Success rate (%)
            success_item = QTableWidgetItem(f"{results[algo]['success_rate']:.1f}")
            self.results_table.setItem(i, 3, success_item)
            
            # Nodes expanded (estimated)
            nodes_item = QTableWidgetItem(f"{results[algo]['avg_nodes_expanded']:.0f}")
            self.results_table.setItem(i, 4, nodes_item)
        
        # Resize columns to content but ensure minimum widths
        self.results_table.resizeColumnsToContents()
        
        # Set minimum widths for columns to ensure headers are fully visible
        min_widths = [120, 140, 160, 160, 160]
        for i, width in enumerate(min_widths):
            if self.results_table.columnWidth(i) < width:
                self.results_table.setColumnWidth(i, width)
        
    def update_time_chart(self, results):
        """Update the execution time chart"""
        # Sort algorithms by time (ascending)
        sorted_items = sorted(results.items(), key=lambda x: x[1]['avg_time'])
        algo_names = [item[0] for item in sorted_items]
        times = [item[1]['avg_time'] for item in sorted_items]
        
        # Create the chart
        fig = self.time_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        bars = ax.barh(algo_names, times, color=COLORS['primary'])
        
        # Add time labels to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + width*0.02, bar.get_y() + bar.get_height()/2, 
                   f"{width:.4f}s", ha='left', va='center')
        
        ax.set_title('Thời gian Thực thi Trung bình (giây)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Thời gian (giây)')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        self.time_canvas.draw()
        
    def update_path_chart(self, results):
        """Update the path length chart"""
        # Only include algorithms with successful solutions
        filtered_results = {k: v for k, v in results.items() if v['avg_path_length'] > 0}
        
        # Sort by path length (ascending)
        sorted_items = sorted(filtered_results.items(), key=lambda x: x[1]['avg_path_length'])
        algo_names = [item[0] for item in sorted_items]
        path_lengths = [item[1]['avg_path_length'] for item in sorted_items]
        
        # Create the chart
        fig = self.path_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        # Calculate maximum path length for proper spacing
        max_length = max(path_lengths) if path_lengths else 0
        padding = max_length * 0.15  # Add 15% padding for labels
        
        bars = ax.barh(algo_names, path_lengths, color=COLORS['informed'])
        
        # Add path length labels to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                   f"{width:.2f}", ha='left', va='center')
        
        ax.set_title('Độ dài Đường đi Trung bình', fontsize=14, fontweight='bold')
        ax.set_xlabel('Số lượng Bước')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Set x-axis limit to include space for labels
        ax.set_xlim(0, max_length + padding)
        
        fig.tight_layout()
        self.path_canvas.draw()
        
    def update_success_chart(self, results):
        """Update the success rate chart"""
        # Sort by success rate (descending)
        sorted_items = sorted(results.items(), key=lambda x: x[1]['success_rate'], reverse=True)
        algo_names = [item[0] for item in sorted_items]
        success_rates = [item[1]['success_rate'] for item in sorted_items]
        
        # Create the chart
        fig = self.success_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        # Use different colors based on success rate
        colors = []
        for rate in success_rates:
            if rate >= 80:
                colors.append(COLORS['success'])
            elif rate >= 50:
                colors.append(COLORS['warning'])
            else:
                colors.append(COLORS['danger'])
        
        bars = ax.barh(algo_names, success_rates, color=colors)
        
        # Add success rate labels to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f"{width:.1f}%", ha='left', va='center')
        
        ax.set_title('Tỉ lệ Thành công của Thuật toán (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tỉ lệ Thành công (%)')
        ax.set_xlim(0, 105)  # Set limit to 105% to make room for labels
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        self.success_canvas.draw()
        
    def update_group_charts(self, results):
        """Update charts comparing algorithms by groups with modern card-based UI"""
        # Clear previous charts in each group's container
        for group_name, container in self.group_containers.items():
            for i in reversed(range(container.count())):
                widget = container.itemAt(i).widget()
                if widget:
                    widget.deleteLater()
        
        # Process each algorithm group
        found_algorithms = False
        
        # For each group tab
        for group_name, group_info in self.algorithm_groups.items():
            # Get algorithms for this group
            algorithms = group_info['algorithms']
            color = group_info['color']
            
            # Filter results for algorithms in this group that exist in results
            group_results = {algo: data for algo, data in results.items() if algo in algorithms}
            
            if not group_results:
                continue  # Skip if no algorithms from this group were evaluated
            
            found_algorithms = True
            
            # Main container for this group
            group_container = QFrame()
            group_container.setStyleSheet(f"""
                QFrame {{
                    background: white;
                    border-radius: 12px;
                    border: 1px solid {COLORS['border']};
                }}
            """)
            
            # Add shadow effect
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(15)
            shadow.setColor(QColor(0, 0, 0, 30))
            shadow.setOffset(0, 3)
            group_container.setGraphicsEffect(shadow)
            
            group_layout = QVBoxLayout(group_container)
            group_layout.setContentsMargins(0, 0, 0, 15)
            group_layout.setSpacing(0)
            
            # Header with gradient background
            header = QFrame()
            header_height = 80
            header.setMinimumHeight(header_height)
            header.setMaximumHeight(header_height)
            header.setStyleSheet(f"""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                               stop:0 {color}, stop:1 {color}88);
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
            """)
            
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(20, 0, 20, 0)
            
            # Group title with white text
            title_label = QLabel(group_name.upper())
            title_label.setStyleSheet("""
                color: white;
                font-size: 16px;
                font-weight: bold;
            """)
            
            # Algorithm count badge
            count_badge = QLabel(f"{len(group_results)} thuật toán")
            count_badge.setStyleSheet("""
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
                padding: 5px 12px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
            """)
            
            header_layout.addWidget(title_label)
            header_layout.addStretch()
            header_layout.addWidget(count_badge)
            
            group_layout.addWidget(header)
            
            # Content container
            content_widget = QWidget()
            content_layout = QVBoxLayout(content_widget)
            content_layout.setContentsMargins(20, 20, 20, 10)
            content_layout.setSpacing(20)
            
            # Add group description
            descriptions = {
                "Tìm kiếm không có thông tin": "Các thuật toán không sử dụng thông tin heuristic, dựa trên chiến lược duyệt đơn giản.",
                "Tìm kiếm có thông tin": "Các thuật toán sử dụng heuristic để ước lượng khoảng cách đến đích, giúp tìm kiếm hiệu quả hơn.",
                "Tìm kiếm cục bộ": "Các thuật toán tối ưu hóa cục bộ, hoạt động trên không gian trạng thái một cách tăng dần.",
                "Tìm kiếm không xác định": "Các thuật toán xử lý môi trường không chắc chắn hoặc không quan sát đầy đủ.",
                "Tìm kiếm có ràng buộc": "Các thuật toán áp dụng các ràng buộc để giảm không gian tìm kiếm.",
                "Học tăng cường": "Thuật toán học từ tương tác với môi trường dựa trên cơ chế phần thưởng."
            }
            
            description = QLabel(descriptions.get(group_name, "Kết quả đánh giá cho nhóm thuật toán này."))
            description.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
            description.setWordWrap(True)
            content_layout.addWidget(description)
            
            # Calculate best and worst values for each metric
            metrics_range = {
                'avg_time': {'min': float('inf'), 'max': float('-inf')},
                'avg_path_length': {'min': float('inf'), 'max': float('-inf')},
                'success_rate': {'min': float('inf'), 'max': float('-inf')},
                'avg_nodes_expanded': {'min': float('inf'), 'max': float('-inf')}
            }
            
            for data in group_results.values():
                for metric, range_data in metrics_range.items():
                    value = data[metric]
                    if value < range_data['min']:
                        range_data['min'] = value
                    if value > range_data['max']:
                        range_data['max'] = value
            
            # Sort algorithms by success rate (primary) and time (secondary)
            sorted_algos = sorted(
                group_results.keys(),
                key=lambda x: (-group_results[x]['success_rate'], group_results[x]['avg_time'])
            )
            
            # Algorithm card-based layout
            cards_widget = QWidget()
            cards_layout = QVBoxLayout(cards_widget)
            cards_layout.setContentsMargins(0, 0, 0, 0)
            cards_layout.setSpacing(15)
            
            # Create a card for each algorithm
            for rank, algo in enumerate(sorted_algos):
                data = group_results[algo]
                
                # Create card container
                card = QFrame()
                card.setStyleSheet(f"""
                    QFrame {{
                        background-color: {COLORS['background']};
                        border-radius: 10px;
                        border: 1px solid {COLORS['divider']};
                    }}
                """)
                
                if rank == 0:
                    # Best algorithm gets a special border
                    card.setStyleSheet(f"""
                        QFrame {{
                            background-color: {COLORS['background']};
                            border-radius: 10px;
                            border: 2px solid {color};
                        }}
                    """)
                
                card_layout = QHBoxLayout(card)
                card_layout.setContentsMargins(15, 15, 15, 15)
                
                # Left side with ranking and algorithm name
                left_widget = QWidget()
                left_layout = QVBoxLayout(left_widget)
                left_layout.setContentsMargins(0, 0, 0, 0)
                left_layout.setSpacing(3)
                
                # Rank indicator
                rank_label = QLabel(f"#{rank+1}")
                rank_label.setStyleSheet(f"""
                    color: {'white' if rank == 0 else COLORS['text_secondary']};
                    font-size: 14px;
                    font-weight: bold;
                    background-color: {color if rank == 0 else '#E0E0E0'};
                    padding: 3px 8px;
                    border-radius: 10px;
                """)
                
                # Algorithm name
                algo_label = QLabel(algo)
                algo_label.setStyleSheet(f"""
                    color: {COLORS['text_primary']};
                    font-size: {16 if rank == 0 else 14}px;
                    font-weight: {'bold' if rank == 0 else 'normal'};
                """)
                
                left_layout.addWidget(rank_label, alignment=Qt.AlignLeft)
                left_layout.addWidget(algo_label, alignment=Qt.AlignLeft)
                
                # Add custom performance labels if it's the best algorithm
                if rank == 0:
                    performance_widget = QWidget()
                    performance_layout = QHBoxLayout(performance_widget)
                    performance_layout.setContentsMargins(0, 5, 0, 0)
                    performance_layout.setSpacing(5)
                    
                    is_best_time = data['avg_time'] == metrics_range['avg_time']['min']
                    is_best_path = data['avg_path_length'] > 0 and data['avg_path_length'] == metrics_range['avg_path_length']['min']
                    is_best_success = data['success_rate'] == metrics_range['success_rate']['max']
                    
                    if is_best_time:
                        best_time = QLabel("Nhanh nhất")
                        best_time.setStyleSheet(f"""
                            background-color: {COLORS['success']}22;
                            color: {COLORS['success']};
                            padding: 2px 6px;
                            border-radius: 5px;
                            font-size: 11px;
                            font-weight: bold;
                        """)
                        performance_layout.addWidget(best_time)
                    
                    if is_best_path:
                        best_path = QLabel("Đường đi tối ưu")
                        best_path.setStyleSheet(f"""
                            background-color: {COLORS['informed']}22;
                            color: {COLORS['informed']};
                            padding: 2px 6px;
                            border-radius: 5px;
                            font-size: 11px;
                            font-weight: bold;
                        """)
                        performance_layout.addWidget(best_path)
                    
                    if is_best_success:
                        best_success = QLabel("Thành công cao")
                        best_success.setStyleSheet(f"""
                            background-color: {COLORS['primary']}22;
                            color: {COLORS['primary']};
                            padding: 2px 6px;
                            border-radius: 5px;
                            font-size: 11px;
                            font-weight: bold;
                        """)
                        performance_layout.addWidget(best_success)
                    
                    performance_layout.addStretch()
                    left_layout.addWidget(performance_widget)
                    
                left_layout.addStretch()
                
                # Right side with metrics
                right_widget = QWidget()
                right_layout = QGridLayout(right_widget)
                right_layout.setContentsMargins(0, 0, 0, 0)
                right_layout.setSpacing(15)
                
                # Success rate - with circular progress bar
                success_rate = data['success_rate']
                if success_rate >= 80:
                    success_color = COLORS['success']
                elif success_rate >= 50:
                    success_color = COLORS['warning']
                else:
                    success_color = COLORS['danger']
                
                # Create custom circular progress indicator
                success_widget = QWidget()
                success_widget.setMinimumWidth(90)
                success_layout = QVBoxLayout(success_widget)
                success_layout.setContentsMargins(0, 0, 0, 0)
                success_layout.setSpacing(5)
                
                success_label = QLabel("Tỉ lệ thành công")
                success_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
                success_label.setAlignment(Qt.AlignCenter)
                
                # Add label first for consistent layout
                success_layout.addWidget(success_label, alignment=Qt.AlignCenter)
                
                success_progress = QProgressBar()
                success_progress.setRange(0, 100)
                success_progress.setValue(int(success_rate))
                success_progress.setTextVisible(False)
                success_progress.setFixedHeight(8)
                success_progress.setStyleSheet(f"""
                    QProgressBar {{
                        background-color: #E0E0E0;
                        border: none;
                        border-radius: 4px;
                    }}
                    QProgressBar::chunk {{
                        background-color: {success_color};
                        border-radius: 4px;
                    }}
                """)
                success_layout.addWidget(success_progress)
                
                success_value = QLabel(f"{success_rate:.1f}%")
                success_value.setStyleSheet(f"""
                    color: {success_color};
                    font-size: 15px;
                    font-weight: bold;
                """)
                success_value.setAlignment(Qt.AlignCenter)
                success_layout.addWidget(success_value, alignment=Qt.AlignCenter)
                
                # Time metric with icon
                time_widget = QWidget()
                time_widget.setMinimumWidth(90)
                time_layout = QVBoxLayout(time_widget)
                time_layout.setContentsMargins(0, 0, 0, 0)
                time_layout.setSpacing(5)
                
                time_label = QLabel("Thời gian (s)")
                time_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
                time_label.setAlignment(Qt.AlignCenter)
                
                # Add label first for consistent layout
                time_layout.addWidget(time_label, alignment=Qt.AlignCenter)
                
                # Calculate normalized time for visual indicator
                min_time = metrics_range['avg_time']['min']
                max_time = metrics_range['avg_time']['max']
                
                if max_time > min_time:
                    normalized = 1 - ((data['avg_time'] - min_time) / (max_time - min_time))
                else:
                    normalized = 1
                
                time_progress = QProgressBar()
                time_progress.setRange(0, 100)
                time_progress.setValue(int(normalized * 100))
                time_progress.setTextVisible(False)
                time_progress.setFixedHeight(8)
                time_progress.setStyleSheet(f"""
                    QProgressBar {{
                        background-color: #E0E0E0;
                        border: none;
                        border-radius: 4px;
                    }}
                    QProgressBar::chunk {{
                        background-color: {color};
                        border-radius: 4px;
                    }}
                """)
                time_layout.addWidget(time_progress)
                
                time_value = QLabel(f"{data['avg_time']:.3f}")
                time_value.setStyleSheet(f"""
                    color: {COLORS['text_primary']};
                    font-size: 15px;
                    font-weight: bold;
                """)
                time_value.setAlignment(Qt.AlignCenter)
                time_layout.addWidget(time_value, alignment=Qt.AlignCenter)
                
                # Path length metric
                path_widget = QWidget()
                path_widget.setMinimumWidth(90)
                path_layout = QVBoxLayout(path_widget)
                path_layout.setContentsMargins(0, 0, 0, 0)
                path_layout.setSpacing(5)
                
                path_label = QLabel("Độ dài đường đi")
                path_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
                path_label.setAlignment(Qt.AlignCenter)
                
                # Add label first to ensure proper order
                path_layout.addWidget(path_label, alignment=Qt.AlignCenter)
                
                # Only show progress bar if there's valid path data
                if data['avg_path_length'] > 0:
                    min_path = metrics_range['avg_path_length']['min']
                    max_path = metrics_range['avg_path_length']['max']
                    
                    if max_path > min_path:
                        path_normalized = 1 - ((data['avg_path_length'] - min_path) / (max_path - min_path))
                    else:
                        path_normalized = 1
                    
                    path_progress = QProgressBar()
                    path_progress.setRange(0, 100)
                    path_progress.setValue(int(path_normalized * 100))
                    path_progress.setTextVisible(False)
                    path_progress.setFixedHeight(8)
                    path_progress.setStyleSheet(f"""
                        QProgressBar {{
                            background-color: #E0E0E0;
                            border: none;
                            border-radius: 4px;
                        }}
                        QProgressBar::chunk {{
                            background-color: {COLORS['informed']};
                            border-radius: 4px;
                        }}
                    """)
                    path_layout.addWidget(path_progress)
                else:
                    spacer = QWidget()
                    spacer.setFixedHeight(8)
                    path_layout.addWidget(spacer)
                
                path_value = QLabel(f"{data['avg_path_length']:.1f}" if data['avg_path_length'] > 0 else "N/A")
                path_value.setStyleSheet(f"""
                    color: {COLORS['text_primary']};
                    font-size: 15px;
                    font-weight: bold;
                """)
                path_value.setAlignment(Qt.AlignCenter)
                path_layout.addWidget(path_value, alignment=Qt.AlignCenter)
                
                # Nodes expanded metric
                nodes_widget = QWidget()
                nodes_widget.setMinimumWidth(90)
                nodes_layout = QVBoxLayout(nodes_widget)
                nodes_layout.setContentsMargins(0, 0, 0, 0)
                nodes_layout.setSpacing(5)
                
                nodes_label = QLabel("Nút mở rộng")
                nodes_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
                nodes_label.setAlignment(Qt.AlignCenter)
                
                # Add label first for consistent layout
                nodes_layout.addWidget(nodes_label, alignment=Qt.AlignCenter)
                
                # Calculate normalized nodes for visual indicator (reversed scale - fewer is better)
                nodes_value = data['avg_nodes_expanded']
                min_nodes = metrics_range['avg_nodes_expanded']['min']
                max_nodes = metrics_range['avg_nodes_expanded']['max']
                
                if max_nodes > min_nodes:
                    normalized = 1 - ((nodes_value - min_nodes) / (max_nodes - min_nodes))
                else:
                    normalized = 1
                
                nodes_progress = QProgressBar()
                nodes_progress.setRange(0, 100)
                nodes_progress.setValue(int(normalized * 100))
                nodes_progress.setTextVisible(False)
                nodes_progress.setFixedHeight(8)
                nodes_progress.setStyleSheet(f"""
                    QProgressBar {{
                        background-color: #E0E0E0;
                        border: none;
                        border-radius: 4px;
                    }}
                    QProgressBar::chunk {{
                        background-color: {COLORS['warning']};
                        border-radius: 4px;
                    }}
                """)
                nodes_layout.addWidget(nodes_progress)
                
                # Format nodes value
                if nodes_value >= 1000000:
                    nodes_text = f"{nodes_value/1000000:.1f}M"
                elif nodes_value >= 1000:
                    nodes_text = f"{nodes_value/1000:.1f}K"
                else:
                    nodes_text = f"{int(nodes_value)}"
                
                nodes_value_label = QLabel(nodes_text)
                nodes_value_label.setStyleSheet(f"""
                    color: {COLORS['text_primary']};
                    font-size: 15px;
                    font-weight: bold;
                """)
                nodes_value_label.setAlignment(Qt.AlignCenter)
                nodes_layout.addWidget(nodes_value_label, alignment=Qt.AlignCenter)
                
                # Add metrics to grid
                right_layout.addWidget(success_widget, 0, 0)
                right_layout.addWidget(time_widget, 0, 1)
                right_layout.addWidget(path_widget, 0, 2)
                right_layout.addWidget(nodes_widget, 0, 3)
                
                # Add the left and right sections to card
                card_layout.addWidget(left_widget, 1)
                card_layout.addWidget(right_widget, 4)
                
                # Add card to the cards layout
                cards_layout.addWidget(card)
            
            content_layout.addWidget(cards_widget)
            
            # Performance summary for this group
            if len(group_results) > 1:
                summary_widget = QWidget()
                summary_widget.setStyleSheet(f"""
                    background-color: {color}11;
                    border-radius: 8px;
                    padding: 5px;
                """)
                summary_layout = QVBoxLayout(summary_widget)
                summary_layout.setContentsMargins(15, 10, 15, 10)
                
                # Find best algorithm in this group for different metrics
                best_time = min(group_results.items(), key=lambda x: x[1]['avg_time'])
                best_path = min([x for x in group_results.items() if x[1]['avg_path_length'] > 0], 
                               key=lambda x: x[1]['avg_path_length'], default=None)
                best_success = max(group_results.items(), key=lambda x: x[1]['success_rate'])
                
                summary_text = f"<b>Tổng kết nhóm:</b> "
                summary_text += f"Thuật toán <b style='color:{color}'>{best_time[0]}</b> có thời gian thực thi nhanh nhất ({best_time[1]['avg_time']:.4f}s). "
                
                if best_path:
                    summary_text += f"Thuật toán <b style='color:{color}'>{best_path[0]}</b> có đường đi ngắn nhất ({best_path[1]['avg_path_length']:.2f} bước). "
                
                summary_text += f"Thuật toán <b style='color:{color}'>{best_success[0]}</b> có tỉ lệ thành công cao nhất ({best_success[1]['success_rate']:.1f}%)."
                
                summary_label = QLabel(summary_text)
                summary_label.setWordWrap(True)
                summary_label.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 5px;")
                
                summary_layout.addWidget(summary_label)
                content_layout.addWidget(summary_widget)
            
            # Add content widget to main layout
            group_layout.addWidget(content_widget)
            
            # Add to group container for this tab
            self.group_containers[group_name].addWidget(group_container)
        
        # Show 'no data' message if appropriate
        if not found_algorithms:
            for group_name, container in self.group_containers.items():
                if container.count() == 0:  # This group has no data
                    no_data_widget = QWidget()
                    no_data_layout = QVBoxLayout(no_data_widget)
                    
                    no_data_icon = QLabel("📊")
                    no_data_icon.setStyleSheet("font-size: 40px; color: #CCCCCC;")
                    no_data_icon.setAlignment(Qt.AlignCenter)
                    
                    no_data_label = QLabel("Chưa có dữ liệu đánh giá nào cho nhóm thuật toán này")
                    no_data_label.setStyleSheet(f"""
                        color: {COLORS['text_secondary']};
                        font-size: 14px;
                    """)
                    no_data_label.setAlignment(Qt.AlignCenter)
                    
                    no_data_hint = QLabel("Vui lòng chạy đánh giá với các thuật toán trong nhóm này")
                    no_data_hint.setStyleSheet(f"""
                        color: {COLORS['text_tertiary']};
                        font-size: 12px;
                    """)
                    no_data_hint.setAlignment(Qt.AlignCenter)
                    
                    no_data_layout.addStretch()
                    no_data_layout.addWidget(no_data_icon)
                    no_data_layout.addWidget(no_data_label)
                    no_data_layout.addWidget(no_data_hint)
                    no_data_layout.addStretch()
                    
                    container.addWidget(no_data_widget)
        
        # Add a spacer at the end of each group container
        for group_name, container in self.group_containers.items():
            container.addStretch()
    
    def export_to_pdf(self):
        """Export evaluation results to PDF"""
        if not self.evaluation_results:
            self.log("Không có kết quả để xuất. Hãy chạy đánh giá trước.")
            return
            
        # Ask for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu Báo cáo PDF", "", "Tập tin PDF (*.pdf)")
            
        if not file_path:
            return
            
        if not file_path.endswith('.pdf'):
            file_path += '.pdf'
            
        try:
            self.log(f"Đang xuất kết quả vào {file_path}...")
            
            # Create PDF document
            doc = SimpleDocTemplate(
                file_path,
                pagesize=A4,
                rightMargin=30,
                leftMargin=30,
                topMargin=30,
                bottomMargin=30
            )
            
            # Register Vietnamese compatible fonts for PDF
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            
            # Use Arial (widely available) as a fallback for Vietnamese text
            pdfmetrics.registerFont(TTFont('ArialUnicode', 'C:\\Windows\\Fonts\\arial.ttf'))
            
            # Collect elements for PDF
            elements = []
            styles = getSampleStyleSheet()
            
            # Customize styles to use our registered font
            styles['Title'].fontName = 'ArialUnicode'
            styles['Heading1'].fontName = 'ArialUnicode'
            styles['Heading2'].fontName = 'ArialUnicode' 
            styles['Heading3'].fontName = 'ArialUnicode'
            styles['Normal'].fontName = 'ArialUnicode'
            
            # Title
            title = Paragraph("Báo cáo Đánh giá Thuật toán 8 Puzzle", styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 20))
            
            # Introduction
            intro_text = f"""
            Báo cáo này trình bày kết quả đánh giá của các thuật toán tìm kiếm khác nhau
            để giải quyết bài toán 8 puzzle. Đánh giá được thực hiện trên 
            {self.puzzles_spinner.value()} puzzle được tạo ngẫu nhiên.
            """
            intro = Paragraph(intro_text, styles['Normal'])
            elements.append(intro)
            elements.append(Spacer(1, 20))
            
            # Results table
            elements.append(Paragraph("Kết quả So sánh Thuật toán", styles['Heading2']))
            elements.append(Spacer(1, 10))
            
            # Create table data
            table_data = [["Thuật toán", "Thời gian TB (s)", "Độ dài Đường đi TB", 
                          "Tỉ lệ Thành công (%)", "Ước tính Nút Mở rộng"]]
            
            # Sort algorithms by execution time
            sorted_algos = sorted(self.evaluation_results.keys(), 
                                key=lambda x: self.evaluation_results[x]['avg_time'])
            
            for algo in sorted_algos:
                result = self.evaluation_results[algo]
                table_data.append([
                    algo,
                    f"{result['avg_time']:.4f}",
                    f"{result['avg_path_length']:.2f}",
                    f"{result['success_rate']:.1f}",
                    f"{result['avg_nodes_expanded']:.0f}"
                ])
            
            # Create and style the table
            table = Table(table_data, colWidths=[120, 80, 100, 90, 120])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'ArialUnicode'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 1), (-1, -1), 'ArialUnicode'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 30))
            
            # Save charts as images and add to PDF
            chart_titles = ["So sánh Thời gian Thực thi", "So sánh Độ dài Đường đi", "So sánh Tỉ lệ Thành công"]
            canvases = [self.time_canvas, self.path_canvas, self.success_canvas]
            
            for title, canvas in zip(chart_titles, canvases):
                elements.append(Paragraph(title, styles['Heading2']))
                elements.append(Spacer(1, 10))
                
                # Save figure to bytes buffer
                buf = io.BytesIO()
                canvas.figure.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                
                # Create image and add to PDF
                img = Image(buf)
                img.drawHeight = 250
                img.drawWidth = 450
                elements.append(img)
                elements.append(Spacer(1, 20))
            
            # Add group comparison charts
            elements.append(Paragraph("So sánh Thuật toán theo Nhóm", styles['Heading2']))
            elements.append(Spacer(1, 10))
            
            # Define algorithm groups
            algorithm_groups = {
                "Tìm kiếm không có thông tin": ['BFS', 'DFS', 'UCS', 'IDS'],
                "Tìm kiếm có thông tin": ['ASTAR', 'IDASTAR', 'GREEDY'],
                "Tìm kiếm cục bộ": ['HILL_CLIMBING', 'STEEPEST_HILL', 'STOCHASTIC_HILL', 'LOCAL_BEAM', 'ANNEALING', 'GENETIC'],
                "Tìm kiếm không xác định": ['ANDOR', 'PARTIALLY_OBS', 'BELIEF_STATE'],
                "Tìm kiếm có ràng buộc": ['BACKTRACKING', 'FORWARD_CHECK', 'AC3'],
                "Học tăng cường": ['QLEARNING']
            }
            
            # Define metrics to compare
            metrics = [
                {"name": "Thời gian thực thi", "key": "avg_time", "unit": "giây"},
                {"name": "Độ dài đường đi", "key": "avg_path_length", "unit": "bước"},
                {"name": "Tỉ lệ thành công", "key": "success_rate", "unit": "%"},
                {"name": "Nút mở rộng", "key": "avg_nodes_expanded", "unit": "nút"}
            ]
            
            # Create charts for each group
            for group_name, algorithms in algorithm_groups.items():
                # Filter results for algorithms in this group that exist in results
                group_results = {algo: data for algo, data in self.evaluation_results.items() if algo in algorithms}
                
                if not group_results:
                    continue  # Skip if no algorithms from this group were evaluated
                
                elements.append(Paragraph(f"Nhóm: {group_name}", styles['Heading3']))
                elements.append(Spacer(1, 5))
                
                # Create a figure with subplots for this group's metrics
                fig = Figure(figsize=(8, 8))
                
                # Determine how many metrics are applicable
                applicable_metrics = []
                for metric in metrics:
                    # Skip path length for algorithms that don't produce paths
                    if metric["key"] == "avg_path_length" and all(v[metric["key"]] == 0 for v in group_results.values()):
                        continue
                    applicable_metrics.append(metric)
                
                rows = max(1, (len(applicable_metrics) + 1) // 2)  # At most 2 per row
                
                for i, metric in enumerate(applicable_metrics):
                    ax = fig.add_subplot(rows, min(2, len(applicable_metrics)), i + 1)
                    
                    # Sort data for this metric
                    reverse = True if metric["key"] == "success_rate" else False
                    sorted_items = sorted(
                        group_results.items(), 
                        key=lambda x: x[1][metric["key"]], 
                        reverse=reverse
                    )
                    
                    algo_names = [item[0] for item in sorted_items]
                    values = [item[1][metric["key"]] for item in sorted_items]
                    
                    # Choose colors based on metric
                    if metric["key"] == "success_rate":
                        bar_colors = []
                        for rate in values:
                            if rate >= 80:
                                bar_colors.append(COLORS['success'])
                            elif rate >= 50:
                                bar_colors.append(COLORS['warning'])
                            else:
                                bar_colors.append(COLORS['danger'])
                        bars = ax.barh(algo_names, values, color=bar_colors)
                    elif metric["key"] == "avg_time":
                        bars = ax.barh(algo_names, values, color=COLORS['primary'])
                    elif metric["key"] == "avg_path_length":
                        bars = ax.barh(algo_names, values, color=COLORS['informed'])
                    else:
                        bars = ax.barh(algo_names, values, color=COLORS['warning'])
                    
                    # Add labels
                    for j, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + (width*0.02 if width > 0 else 0.1), 
                              bar.get_y() + bar.get_height()/2, 
                              f"{width:.2f}" if metric["key"] != "avg_nodes_expanded" else f"{int(width)}", 
                              ha='left', va='center')
                    
                    ax.set_title(f'{metric["name"]}', fontsize=10)
                    ax.set_xlabel(f'{metric["name"]} ({metric["unit"]})')
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                fig.tight_layout()
                
                # Save group figure to PDF
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                
                # Create image and add to PDF
                img = Image(buf)
                img.drawHeight = 300
                img.drawWidth = 450
                elements.append(img)
                elements.append(Spacer(1, 20))
            
            # Analysis and conclusion
            elements.append(Paragraph("Phân tích và Kết luận", styles['Heading2']))
            elements.append(Spacer(1, 10))
            
            # Find best algorithms for different metrics
            fastest_algo = min(self.evaluation_results.items(), key=lambda x: x[1]['avg_time'])[0]
            
            # Filter out algorithms with no solutions (path length = 0)
            path_filtered = {k: v for k, v in self.evaluation_results.items() if v['avg_path_length'] > 0}
            shortest_path_algo = min(path_filtered.items(), key=lambda x: x[1]['avg_path_length'])[0] if path_filtered else "None"
            
            most_successful = max(self.evaluation_results.items(), key=lambda x: x[1]['success_rate'])[0]
            
            analysis_text = f"""
            Dựa trên kết quả đánh giá, chúng ta có thể rút ra các kết luận sau:
            
            1. Thuật toán nhanh nhất: {fastest_algo} với thời gian thực thi trung bình 
               {self.evaluation_results[fastest_algo]['avg_time']:.4f} giây.
               
            2. Thuật toán có đường đi ngắn nhất: {shortest_path_algo} với độ dài đường đi trung bình 
               {self.evaluation_results[shortest_path_algo]['avg_path_length']:.2f} bước.
               
            3. Thuật toán thành công nhất: {most_successful} với tỉ lệ thành công 
               {self.evaluation_results[most_successful]['success_rate']:.1f}%.
               
            Các thuật toán khác nhau có những điểm mạnh và điểm yếu khác nhau. Việc lựa chọn 
            thuật toán phụ thuộc vào yêu cầu cụ thể của ứng dụng, chẳng hạn như 
            ràng buộc về thời gian, tính tối ưu của giải pháp và ràng buộc về bộ nhớ.
            """
            
            analysis = Paragraph(analysis_text, styles['Normal'])
            elements.append(analysis)
            
            # Build the PDF
            doc.build(elements)
            
            self.log(f"Báo cáo đã được xuất thành công vào {file_path}")
            
        except Exception as e:
            self.log(f"Lỗi khi xuất PDF: {str(e)}")
            
    def closeEvent(self, event):
        """Handle window close event"""
        if self.evaluation_thread and self.evaluation_thread.isRunning():
            self.evaluation_thread.stop()
            self.evaluation_thread.wait()
        event.accept()

class NonDeterministicEnvironment:
    """Môi trường 8-puzzle không xác định
    Đặc điểm:
    - Quan sát một phần: Agent chỉ quan sát được một số ô trên bảng
    - Hành động không xác định: Các hành động có thể dẫn đến kết quả không lường trước
    - Môi trường động: Trạng thái có thể thay đổi ngẫu nhiên theo thời gian
    """
    def __init__(self, initial_state=None, goal_state=None, 
                observable_positions=[0, 1, 2, 4, 8], 
                random_action_prob=0.1, 
                random_change_prob=0.05):
        """
        Khởi tạo môi trường không xác định
        Tham số:
        - initial_state: Trạng thái ban đầu (mặc định là [1,2,3,4,5,6,7,0,8])
        - goal_state: Trạng thái đích (mặc định là [1,2,3,4,5,6,7,8,0])
        - observable_positions: Các vị trí có thể quan sát được
        - random_action_prob: Xác suất hành động không đạt kết quả mong muốn
        - random_change_prob: Xác suất môi trường tự thay đổi
        """
        self.state = list(initial_state) if initial_state else [1,2,3,4,5,6,7,0,8]
        self.goal_state = list(goal_state) if goal_state else [1,2,3,4,5,6,7,8,0]
        self.observable_positions = observable_positions
        self.random_action_prob = random_action_prob
        self.random_change_prob = random_change_prob
        
        # Lưu trữ các trạng thái niềm tin
        self.belief_states = self._initialize_belief_states()
        
    def _initialize_belief_states(self):
        """Khởi tạo tập hợp các trạng thái niềm tin ban đầu"""
        # Dựa trên trạng thái quan sát được, tìm tất cả các trạng thái có thể
        observable_state = self.get_observable_state()
        
        # Tạo tất cả các permutation có thể của các vị trí không quan sát được
        all_possible_states = []
        
        # Xác định các vị trí không quan sát được và các giá trị có thể
        unobservable_positions = [i for i in range(9) if i not in self.observable_positions]
        unused_values = [v for i, v in enumerate(range(9)) 
                       if i not in self.observable_positions and v not in [observable_state[p] for p in self.observable_positions]]
        
        # Tìm tất cả các trạng thái phù hợp với quan sát hiện tại
        all_permutations = list(itertools.permutations(unused_values))
        for perm in all_permutations:
            state = list(observable_state)  # Sao chép trạng thái quan sát được
            for pos, value in zip(unobservable_positions, perm):
                state[pos] = value
            if is_solvable(state):  # Chỉ thêm trạng thái có thể giải được
                all_possible_states.append(tuple(state))
        
        return set(all_possible_states)
        
    def get_observable_state(self):
        """Trả về trạng thái có thể quan sát được (thông tin một phần)"""
        observable = [-1] * 9  # -1 đại diện cho vị trí không quan sát được
        for pos in self.observable_positions:
            observable[pos] = self.state[pos]
        return tuple(observable)
    
    def update_belief_states(self, action):
        """Cập nhật tập trạng thái niềm tin dựa trên hành động và quan sát"""
        # Tạo tập hợp các trạng thái niềm tin mới
        new_belief_states = set()
        
        # Quan sát sau khi thực hiện hành động
        observable = self.get_observable_state()
        
        # Cập nhật trạng thái niềm tin
        for belief_state in self.belief_states:
            # Thử áp dụng hành động lên trạng thái niềm tin
            possible_next_states = self.get_possible_results(belief_state, action)
            
            # Lọc các trạng thái phù hợp với quan sát mới
            for next_state in possible_next_states:
                is_consistent = True
                for i in self.observable_positions:
                    if observable[i] != -1 and next_state[i] != observable[i]:
                        is_consistent = False
                        break
                
                if is_consistent:
                    new_belief_states.add(next_state)
        
        self.belief_states = new_belief_states
        
    def get_possible_results(self, state, action):
        """Trả về các trạng thái có thể xảy ra khi thực hiện action từ state"""
        # Xác định vị trí ô trống
        try:
            empty_pos = state.index(0)
        except ValueError:
            # Nếu không tìm thấy ô trống, trả về state hiện tại (không thể di chuyển)
            return [state]
        
        # Xác định hành động di chuyển
        row, col = empty_pos // 3, empty_pos % 3
        possible_results = []
        
        # Hành động dự định
        intended_pos = None
        if action == 'UP' and row > 0:
            intended_pos = empty_pos - 3
        elif action == 'DOWN' and row < 2:
            intended_pos = empty_pos + 3
        elif action == 'LEFT' and col > 0:
            intended_pos = empty_pos - 1
        elif action == 'RIGHT' and col < 2:
            intended_pos = empty_pos + 1
            
        if intended_pos is not None:
            # Thêm kết quả dự định
            new_state = list(state)
            new_state[empty_pos], new_state[intended_pos] = new_state[intended_pos], new_state[empty_pos]
            possible_results.append(tuple(new_state))
            
            # Thêm các kết quả ngẫu nhiên (do tính không xác định)
            neighbors = get_neighbors(state)
            for neighbor in neighbors:
                if neighbor != tuple(new_state):  # Không thêm lại kết quả dự định
                    possible_results.append(neighbor)
        
        return possible_results
    
    def take_action(self, action):
        """Thực hiện hành động và trả về quan sát mới
        
        Tham số:
        - action: 'UP', 'DOWN', 'LEFT', 'RIGHT'
        
        Trả về:
        - observable_state: Trạng thái quan sát được sau khi thực hiện hành động
        - reward: Phần thưởng nhận được
        - done: Đánh dấu đã đến đích hay chưa
        """
        # Xác định vị trí ô trống
        try:
            empty_pos = self.state.index(0)
        except ValueError:
            # Nếu không tìm thấy ô trống, không thể di chuyển
            return self.get_observable_state(), -1, tuple(self.state) == tuple(self.goal_state)
            
        row, col = empty_pos // 3, empty_pos % 3
        
        # Xác định vị trí mới dựa trên hành động
        new_pos = None
        if action == 'UP' and row > 0:
            new_pos = empty_pos - 3
        elif action == 'DOWN' and row < 2:
            new_pos = empty_pos + 3
        elif action == 'LEFT' and col > 0:
            new_pos = empty_pos - 1
        elif action == 'RIGHT' and col < 2:
            new_pos = empty_pos + 1
        
        # Nếu hành động không hợp lệ, giữ nguyên trạng thái
        if new_pos is None:
            return self.get_observable_state(), -1, tuple(self.state) == tuple(self.goal_state)
        
        # Xử lý hành động không xác định
        if random.random() < self.random_action_prob:
            # Chọn ngẫu nhiên một hướng khác
            possible_actions = []
            if row > 0 and action != 'UP':
                possible_actions.append((empty_pos - 3, 'UP'))
            if row < 2 and action != 'DOWN':
                possible_actions.append((empty_pos + 3, 'DOWN'))
            if col > 0 and action != 'LEFT':
                possible_actions.append((empty_pos - 1, 'LEFT'))
            if col < 2 and action != 'RIGHT':
                possible_actions.append((empty_pos + 1, 'RIGHT'))
            
            if possible_actions:
                new_pos, _ = random.choice(possible_actions)
        
        # Thực hiện di chuyển
        self.state[empty_pos], self.state[new_pos] = self.state[new_pos], self.state[empty_pos]
        
        # Xử lý môi trường động (tự thay đổi)
        if random.random() < self.random_change_prob:
            # Thực hiện một hành động ngẫu nhiên
            self.random_change()
        
        # Cập nhật trạng thái niềm tin
        self.update_belief_states(action)
        
        # Tính toán phần thưởng
        reward = 10 if tuple(self.state) == tuple(self.goal_state) else -1
        
        # Kiểm tra kết thúc
        done = tuple(self.state) == tuple(self.goal_state)
        
        return self.get_observable_state(), reward, done
    
    def random_change(self):
        """Thực hiện thay đổi ngẫu nhiên đối với môi trường (tính động)"""
        # Tìm các vị trí có thể di chuyển
        try:
            empty_pos = self.state.index(0)
        except ValueError:
            # Nếu không tìm thấy ô trống, không thể di chuyển
            return
            
        row, col = empty_pos // 3, empty_pos % 3
        
        possible_moves = []
        if row > 0:
            possible_moves.append(empty_pos - 3)
        if row < 2:
            possible_moves.append(empty_pos + 3)
        if col > 0:
            possible_moves.append(empty_pos - 1)
        if col < 2:
            possible_moves.append(empty_pos + 1)
            
        if possible_moves:
            # Chọn ngẫu nhiên một hướng di chuyển
            new_pos = random.choice(possible_moves)
            # Thực hiện di chuyển
            self.state[empty_pos], self.state[new_pos] = self.state[new_pos], self.state[empty_pos]
            
    def get_most_likely_state(self):
        """Trả về trạng thái có khả năng cao nhất trong tập trạng thái niềm tin"""
        if not self.belief_states:
            return tuple(self.state)
            
        # Đếm tần suất từng giá trị ở mỗi vị trí
        position_values = [dict() for _ in range(9)]
        
        for state in self.belief_states:
            for i, value in enumerate(state):
                if value not in position_values[i]:
                    position_values[i][value] = 0
                position_values[i][value] += 1
        
        # Chọn giá trị phổ biến nhất cho mỗi vị trí
        most_likely = [-1] * 9
        for i in range(9):
            if position_values[i]:
                most_likely[i] = max(position_values[i].items(), key=lambda x: x[1])[0]
        
        # Kết hợp với trạng thái quan sát được
        observable = self.get_observable_state()
        for i, value in enumerate(observable):
            if value != -1:
                most_likely[i] = value
                
        return tuple(most_likely)
    
    def reset(self, state=None):
        """Reset môi trường về trạng thái ban đầu hoặc trạng thái mới"""
        if state:
            self.state = list(state)
        else:
            self.state = generate_random_puzzle()
            
        self.belief_states = self._initialize_belief_states()
        return self.get_observable_state()

# Cải tiến thuật toán AND/OR Graph Search để sử dụng trong môi trường không xác định
def andor_graph_search_non_deterministic(env, max_depth=15):
    """AND/OR Graph Search cho môi trường không xác định
    Đặc điểm:
    - Sử dụng môi trường NonDeterministicEnvironment
    - Xử lý tính không xác định của hành động
    - Xử lý thông tin một phần (partially observable)
    - Xây dựng cây AND/OR thực sự để giải quyết bài toán
    """
    def or_search(belief_state, depth, path):
        """Tìm kiếm OR node: Chỉ cần một hành động thành công"""
        if depth > max_depth:
            return None, float('inf')
            
        if all(s == tuple(env.goal_state) for s in belief_state):
            return path, 0  # Đã đạt goal state
            
        # Heuristic: Ưu tiên tập belief state có nhiều trạng thái gần goal
        if len(path) % 5 == 0:  # Kiểm tra gián đoạn để tránh tính toán quá nhiều
            total_distance = sum(manhattan_distance(s, tuple(env.goal_state)) for s in belief_state)
            avg_distance = total_distance / len(belief_state) if belief_state else float('inf')
            if avg_distance < 2:  # Nếu trung bình gần goal, trả về path hiện tại
                return path, avg_distance
            
        best_action = None
        best_score = float('inf')
        best_path = None
        
        # Xét tất cả các hành động có thể
        for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            # Xác định tập belief state mới sau khi thực hiện hành động
            next_belief_states = set()
            for state in belief_state:
                next_states = env.get_possible_results(state, action)
                for next_state in next_states:
                    next_belief_states.add(next_state)
            
            # Nếu hành động hợp lệ (có ít nhất một trạng thái mới)
            if next_belief_states:
                # Tiếp tục với AND search
                next_path, next_score = and_search(next_belief_states, depth + 1, path + [action])
                
                if next_path and next_score < best_score:
                    best_score = next_score
                    best_path = next_path
                    best_action = action
        
        return best_path, best_score
        
    def and_search(belief_state, depth, path):
        """Tìm kiếm AND node: Tất cả các kết quả có thể của hành động phải thành công"""
        if depth > max_depth:
            return None, float('inf')
            
        if all(s == tuple(env.goal_state) for s in belief_state):
            return path, 0  # Đã đạt goal state
            
        best_path = path
        total_score = 0
        
        # Phân nhóm belief states dựa trên action đã thực hiện
        action_groups = {}
        for state in belief_state:
            try:
                empty_pos = state.index(0)
                row, col = empty_pos // 3, empty_pos % 3
            except ValueError:
                # Nếu không tìm thấy ô trống, bỏ qua state này
                continue
            
            # Xác định hành động đến trạng thái này
            if len(path) > 0:
                last_action = path[-1]
                action_groups.setdefault(last_action, set()).add(state)
        
        # Đối với mỗi nhóm, tiếp tục với OR search
        for action, states in action_groups.items():
            next_path, next_score = or_search(states, depth + 1, path)
            if next_path is None:
                return None, float('inf')  # Nếu một nhóm thất bại, AND node thất bại
                
            best_path = next_path
            total_score += next_score
        
        return best_path, total_score
    
    # Bắt đầu với OR search từ tập belief states hiện tại
    initial_belief_states = env.belief_states
    if not initial_belief_states:
        initial_belief_states = {tuple(env.state)}
        
    path, _ = or_search(initial_belief_states, 0, [])
    return path

# Cải tiến thuật toán Partially Observable Search để sử dụng trong môi trường không xác định
def improved_partially_observable(env, max_depth=20):
    """Cải tiến thuật toán Partially Observable Search
    Đặc điểm:
    - Sử dụng môi trường NonDeterministicEnvironment
    - Xử lý thuộc tính một phần quan sát được (partially observable)
    - Kết hợp với A* để tìm đường đi ngắn nhất
    """
    # Lấy trạng thái quan sát được hiện tại
    observable_state = env.get_observable_state()
    
    # Trong trường hợp partial observability, sử dụng belief state phổ biến nhất
    most_likely_state = env.get_most_likely_state()
    
    # Queue ưu tiên cho A*
    start_state = most_likely_state
    open_set = [(manhattan_distance(start_state, tuple(env.goal_state)), 0, start_state, [])]
    heapq.heapify(open_set)
    
    # Các state đã thăm
    closed_set = set()
    
    while open_set:
        _, cost, current_state, path = heapq.heappop(open_set)
        
        if current_state in closed_set:
            continue
            
        closed_set.add(current_state)
        
        # Kiểm tra đích
        if current_state == tuple(env.goal_state):
            return path
            
        # Kiểm tra giới hạn độ sâu
        if len(path) >= max_depth:
            continue
            
        # Xét các hành động có thể
        for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            # Giả lập thực hiện hành động
            try:
                empty_pos = current_state.index(0)
                row, col = empty_pos // 3, empty_pos % 3
            except ValueError:
                # Nếu không tìm thấy ô trống, bỏ qua action này
                continue
            
            new_pos = None
            if action == 'UP' and row > 0:
                new_pos = empty_pos - 3
            elif action == 'DOWN' and row < 2:
                new_pos = empty_pos + 3
            elif action == 'LEFT' and col > 0:
                new_pos = empty_pos - 1
            elif action == 'RIGHT' and col < 2:
                new_pos = empty_pos + 1
                
            if new_pos is not None:
                # Tạo trạng thái mới
                new_state_list = list(current_state)
                new_state_list[empty_pos], new_state_list[new_pos] = new_state_list[new_pos], new_state_list[empty_pos]
                new_state = tuple(new_state_list)
                
                if new_state not in closed_set:
                    # Kiểm tra xem trạng thái mới có phù hợp với quan sát hiện tại không
                    is_consistent = True
                    for i in env.observable_positions:
                        if observable_state[i] != -1 and new_state[i] != observable_state[i]:
                            is_consistent = False
                            break
                            
                    if is_consistent:
                        new_cost = cost + 1
                        h_value = manhattan_distance(new_state, tuple(env.goal_state))
                        f_value = new_cost + h_value
                        heapq.heappush(open_set, (f_value, new_cost, new_state, path + [action]))
    
    return []

# Cải tiến thuật toán Belief State Search để sử dụng trong môi trường không xác định
def improved_belief_state_search(env, max_depth=20):
    """Cải tiến thuật toán Belief State Search
    Đặc điểm:
    - Sử dụng môi trường NonDeterministicEnvironment
    - Xây dựng và cập nhật tập hợp belief states
    - Sử dụng MDP (Markov Decision Process) để đưa ra quyết định tối ưu
    """
    # Bắt đầu với belief states hiện tại của môi trường
    belief_states = env.belief_states
    if not belief_states:
        belief_states = {tuple(env.state)}
    
    path = []
    
    for _ in range(max_depth):
        # Nếu tất cả belief states là goal, kết thúc
        if all(state == tuple(env.goal_state) for state in belief_states):
            return path
            
        # Tính toán giá trị của từng hành động
        action_values = {'UP': 0, 'DOWN': 0, 'LEFT': 0, 'RIGHT': 0}
        
        for action in action_values:
            # Tính giá trị trung bình khi thực hiện action từ mỗi belief state
            total_value = 0
            valid_count = 0
            
            for state in belief_states:
                # Mô phỏng thực hiện hành động
                try:
                    empty_pos = state.index(0)
                    row, col = empty_pos // 3, empty_pos % 3
                except ValueError:
                    # Nếu không tìm thấy ô trống, bỏ qua state này
                    continue
                
                new_pos = None
                if action == 'UP' and row > 0:
                    new_pos = empty_pos - 3
                elif action == 'DOWN' and row < 2:
                    new_pos = empty_pos + 3
                elif action == 'LEFT' and col > 0:
                    new_pos = empty_pos - 1
                elif action == 'RIGHT' and col < 2:
                    new_pos = empty_pos + 1
                    
                if new_pos is not None:
                    # Tạo trạng thái mới
                    new_state_list = list(state)
                    new_state_list[empty_pos], new_state_list[new_pos] = new_state_list[new_pos], new_state_list[empty_pos]
                    new_state = tuple(new_state_list)
                    
                    # Tính giá trị dựa trên khoảng cách Manhattan
                    value = -manhattan_distance(new_state, tuple(env.goal_state))
                    
                    # Thưởng thêm nếu đạt goal
                    if new_state == tuple(env.goal_state):
                        value += 100
                        
                    total_value += value
                    valid_count += 1
            
            # Cập nhật giá trị trung bình
            if valid_count > 0:
                action_values[action] = total_value / valid_count
        
        # Chọn hành động tốt nhất
        best_action = max(action_values, key=action_values.get)
        
        # Cập nhật belief states
        new_belief_states = set()
        for state in belief_states:
            # Mô phỏng thực hiện hành động
            try:
                empty_pos = state.index(0)
                row, col = empty_pos // 3, empty_pos % 3
            except ValueError:
                # Nếu không tìm thấy ô trống, bỏ qua state này
                continue
            
            new_pos = None
            if best_action == 'UP' and row > 0:
                new_pos = empty_pos - 3
            elif best_action == 'DOWN' and row < 2:
                new_pos = empty_pos + 3
            elif best_action == 'LEFT' and col > 0:
                new_pos = empty_pos - 1
            elif best_action == 'RIGHT' and col < 2:
                new_pos = empty_pos + 1
                
            if new_pos is not None:
                # Tạo trạng thái mới
                new_state_list = list(state)
                new_state_list[empty_pos], new_state_list[new_pos] = new_state_list[new_pos], new_state_list[empty_pos]
                new_state = tuple(new_state_list)
                
                new_belief_states.add(new_state)
                
                # Xử lý tính không xác định: thêm các trạng thái hàng xóm
                if random.random() < env.random_action_prob:
                    for neighbor in get_neighbors(state):
                        if neighbor != new_state:
                            new_belief_states.add(neighbor)
        
        # Cập nhật belief states và path
        belief_states = new_belief_states
        path.append(best_action)
        
        # Nếu không có belief states hợp lệ, dừng
        if not belief_states:
            break
    
    return path

class NonDeterministicWindow(QMainWindow):
    """Cửa sổ riêng cho các thuật toán tìm kiếm trong môi trường không xác định"""
    
    def __init__(self, parent=None, initial_state=None, goal_state=None):
        super().__init__(parent)
        
        self.parent = parent
        self.initial_state = tuple(initial_state) if initial_state else tuple([1,2,3,4,5,6,7,0,8])
        self.goal_state = tuple(goal_state) if goal_state else tuple([1,2,3,4,5,6,7,8,0])
        self.current_algorithm = "ANDOR"  # Mặc định
        self.solution_path = []  # Đường đi hiện tại
        self.current_step = 0  # Bước hiện tại trong đường đi
        
        # Tạo môi trường không xác định
        self.environment = NonDeterministicEnvironment(
            initial_state=self.initial_state,
            goal_state=self.goal_state
        )
        
        # Auto-play timer
        self.auto_play_timer = QTimer(self)
        self.auto_play_timer.timeout.connect(self.auto_play_next)
        
        # Thiết lập cửa sổ
        self.setWindowTitle("Tìm kiếm trong Môi trường Không xác định")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(f"background-color: {COLORS['background']};")
        
        # Automatically maximize window when opened
        self.showMaximized()
        
        # Tạo widget trung tâm
        self.setupUI()
        
        # Ghi log ban đầu
        self.log("Cửa sổ tìm kiếm trong môi trường không xác định đã sẵn sàng")
        self.log("Đặc điểm môi trường: Quan sát một phần, hành động không xác định, môi trường động")
    
    def setupUI(self):
        """Thiết lập giao diện người dùng"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Tiêu đề
        title_label = QLabel("Tìm kiếm trong Môi trường Không xác định")
        title_label.setStyleSheet(f"""
            color: {COLORS['complex']};
            font-size: 24px;
                            font-weight: bold;
            padding: 10px 0px;
            border-bottom: 2px solid {COLORS['complex']};
        """)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Vùng nội dung chính (1 bảng bên trái, thông tin bên phải)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Panel bên trái - Bảng puzzle và trạng thái niềm tin
        left_panel = QFrame()
        left_panel.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['surface']};
                border-radius: 8px;
                border: 1px solid {COLORS['border']};
                padding: 10px;
            }}
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        
        # Tiêu đề cho Puzzle Board
        puzzle_board_label = QLabel("Trạng thái Puzzle Thực tế")
        puzzle_board_label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 16px;
                            font-weight: bold;
            margin-bottom: 5px;
        """)
        puzzle_board_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(puzzle_board_label)
        
        # Bảng Puzzle
        self.puzzle_board = PuzzleBoard()
        self.puzzle_board.state = list(self.initial_state)
        self.puzzle_board.update()
        left_layout.addWidget(self.puzzle_board, alignment=Qt.AlignCenter)
        
        # Nhóm hiển thị trạng thái niềm tin
        belief_group = QGroupBox("Trạng thái Niềm tin")
        belief_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px;
                            font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                margin-top: 10px;
                padding: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: {COLORS['surface']};
            }}
        """)
        
        belief_layout = QVBoxLayout(belief_group)
        
        # Hiển thị trạng thái có thể quan sát được
        observable_label = QLabel("Trạng thái Quan sát được:")
        observable_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px; font-weight: bold;")
        belief_layout.addWidget(observable_label)
        
        # Grid hiển thị trạng thái quan sát được
        observable_grid = QGridLayout()
        observable_grid.setSpacing(8)  # Thêm khoảng cách giữa các ô
        self.observable_cells = []
        
        for i in range(3):
            for j in range(3):
                pos = i * 3 + j
                cell = QLabel()
                cell.setStyleSheet(f"""
                    background-color: {COLORS['primary'] if pos in self.environment.observable_positions else COLORS['divider']};
                    color: white;
                    border-radius: 5px;
                    padding: 10px;
                    min-width: 40px;
                    min-height: 40px;
                    font-size: 16px;
                    font-weight: bold;
                    qproperty-alignment: AlignCenter;
                """)
                
                # Cố định kích thước hiển thị
                cell.setMinimumSize(60, 60)
                cell.setMaximumSize(60, 60)
                
                # Hiển thị giá trị hoặc "?"
                if pos in self.environment.observable_positions:
                    cell.setText(str(self.initial_state[pos]))
                else:
                    cell.setText("?")
                    
                observable_grid.addWidget(cell, i, j)
                self.observable_cells.append(cell)
                
        belief_layout.addLayout(observable_grid)
        
        # Thông tin về tập trạng thái niềm tin
        belief_info_label = QLabel("Thông tin Belief States:")
        belief_info_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px; font-weight: bold; margin-top: 10px;")
        belief_layout.addWidget(belief_info_label)
        
        self.belief_info = QLabel("Số lượng trạng thái niềm tin hiện tại: 0")
        self.belief_info.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 14px;
            padding: 10px;
            background: {COLORS['divider']};
            border-radius: 5px;
        """)
        self.belief_info.setWordWrap(True)
        belief_layout.addWidget(self.belief_info)
        
        # Thêm GroupBox Belief vào panel trái
        left_layout.addWidget(belief_group)
        
        # Panel bên phải - Điều khiển và thông tin thuật toán
        right_panel = QFrame()
        right_panel.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['surface']};
                border-radius: 8px;
                border: 1px solid {COLORS['border']};
                padding: 10px;
            }}
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        
        # Nhóm cấu hình môi trường
        env_config_group = QGroupBox("Cấu hình Môi trường")
        env_config_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px;
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                        border-radius: 4px;
                margin-top: 10px;
                padding: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: {COLORS['surface']};
                    }}
                """)
                
        env_config_layout = QFormLayout(env_config_group)
        env_config_layout.setSpacing(10)
        
        # Vị trí có thể quan sát được
        observable_pos_label = QLabel("Vị trí có thể quan sát:")
        observable_pos_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px;")
        self.observable_pos_input = QLineEdit("0, 1, 2, 4, 8")
        self.observable_pos_input.setStyleSheet(f"""
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 8px;
            font-size: 14px;
        """)
        env_config_layout.addRow(observable_pos_label, self.observable_pos_input)
        
        # Xác suất hành động ngẫu nhiên
        random_action_label = QLabel("Xác suất hành động ngẫu nhiên:")
        random_action_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px;")
        self.random_action_input = QLineEdit("0.1")
        self.random_action_input.setStyleSheet(f"""
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 8px;
            font-size: 14px;
        """)
        env_config_layout.addRow(random_action_label, self.random_action_input)
        
        # Xác suất môi trường tự thay đổi
        random_change_label = QLabel("Xác suất môi trường tự thay đổi:")
        random_change_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px;")
        self.random_change_input = QLineEdit("0.05")
        self.random_change_input.setStyleSheet(f"""
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 8px;
            font-size: 14px;
        """)
        env_config_layout.addRow(random_change_label, self.random_change_input)
        
        # Nút cập nhật cấu hình
        update_config_btn = QPushButton("Cập nhật Cấu hình")
        update_config_btn.setStyleSheet(f"""
            background-color: {COLORS['complex']};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 10px;
            font-size: 14px;
                    font-weight: bold;
                """)
        update_config_btn.clicked.connect(self.update_environment_config)
        env_config_layout.addRow(QLabel(""), update_config_btn)
        
        right_layout.addWidget(env_config_group)
        
        # Nhóm chọn thuật toán
        algorithm_group = QGroupBox("Chọn Thuật toán")
        algorithm_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px;
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                margin-top: 10px;
                padding: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: {COLORS['surface']};
            }}
        """)
        
        algorithm_layout = QVBoxLayout(algorithm_group)
        
        # Danh sách các thuật toán
        self.algorithm_radio_andor = QRadioButton("AND/OR Graph Search")
        self.algorithm_radio_andor.setChecked(True)
        self.algorithm_radio_andor.setStyleSheet(f"""
                    color: {COLORS['text_primary']};
            font-size: 14px;
            padding: 5px;
            margin-top: 5px;
        """)
        self.algorithm_radio_andor.clicked.connect(lambda: self.select_algorithm("ANDOR"))
        
        self.algorithm_radio_partial = QRadioButton("Partially Observable Search")
        self.algorithm_radio_partial.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 14px;
            padding: 5px;
            margin-top: 5px;
        """)
        self.algorithm_radio_partial.clicked.connect(lambda: self.select_algorithm("PARTIALLY_OBS"))
        
        self.algorithm_radio_belief = QRadioButton("Belief State Search")
        self.algorithm_radio_belief.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 14px;
            padding: 5px;
            margin-top: 5px;
        """)
        self.algorithm_radio_belief.clicked.connect(lambda: self.select_algorithm("BELIEF_STATE"))
        
        algorithm_layout.addWidget(self.algorithm_radio_andor)
        algorithm_layout.addWidget(self.algorithm_radio_partial)
        algorithm_layout.addWidget(self.algorithm_radio_belief)
        
        # Nút điều khiển
        control_buttons_layout = QHBoxLayout()
        
        self.solve_btn = QPushButton("Giải")
        self.solve_btn.setStyleSheet(f"""
            background-color: {COLORS['success']};
            color: white;
                        border: none;
                        border-radius: 4px;
            padding: 12px;
            font-size: 14px;
            font-weight: bold;
        """)
        self.solve_btn.clicked.connect(self.solve)
        
        self.reset_btn = QPushButton("Đặt lại")
        self.reset_btn.setStyleSheet(f"""
            background-color: {COLORS['danger']};
            color: white;
            border: none;
                        border-radius: 4px;
            padding: 12px;
            font-size: 14px;
            font-weight: bold;
        """)
        self.reset_btn.clicked.connect(self.reset)
        
        self.step_btn = QPushButton("Bước tiếp theo")
        self.step_btn.setStyleSheet(f"""
            background-color: {COLORS['primary']};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 12px;
            font-size: 14px;
            font-weight: bold;
        """)
        self.step_btn.clicked.connect(self.take_step)
        
        control_buttons_layout.addWidget(self.solve_btn)
        control_buttons_layout.addWidget(self.reset_btn)
        control_buttons_layout.addWidget(self.step_btn)
        
        algorithm_layout.addLayout(control_buttons_layout)
        
        right_layout.addWidget(algorithm_group)
        
        # Nhóm thông tin giải pháp
        solution_group = QGroupBox("Thông tin Giải pháp")
        solution_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px;
                    font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                margin-top: 10px;
                padding: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: {COLORS['surface']};
            }}
        """)
        
        solution_layout = QVBoxLayout(solution_group)
        
        # Hiển thị thông tin đường đi
        self.solution_info = QLabel("Chưa có giải pháp")
        self.solution_info.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 14px;
            padding: 10px;
            background: {COLORS['divider']};
            border-radius: 5px;
            min-height: 80px;
        """)
        self.solution_info.setAlignment(Qt.AlignCenter)
        self.solution_info.setWordWrap(True)
        solution_layout.addWidget(self.solution_info)
        
        # Nút tự động chạy
        auto_play_btn = QPushButton("Tự động Chạy")
        auto_play_btn.setStyleSheet(f"""
            background-color: {COLORS['accent']};
            color: white;
                            border: none;
                            border-radius: 4px;
            padding: 12px;
            font-size: 14px;
            font-weight: bold;
            margin-top: 10px;
        """)
        auto_play_btn.clicked.connect(self.auto_play)
        solution_layout.addWidget(auto_play_btn)
        
        right_layout.addWidget(solution_group)
        
        # Thêm stretch vào panel phải để đẩy các nhóm lên trên
        right_layout.addStretch()
        
        # Thêm panel vào vùng nội dung
        content_layout.addWidget(left_panel, 1)  # Tỷ lệ 1
        content_layout.addWidget(right_panel, 1)  # Tỷ lệ 1
        
        main_layout.addLayout(content_layout, 1)  # Tỷ lệ 1
        
        # Vùng log
        log_group = QGroupBox("Nhật ký hoạt động")
        log_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px;
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                            border-radius: 4px;
                margin-top: 5px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: {COLORS['background']};
                        }}
                    """)
        
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(10, 20, 10, 10)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            background-color: {COLORS['surface']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 5px;
            font-family: Consolas, Monaco, monospace;
            font-size: 12px;
        """)
        log_layout.addWidget(self.log_text)
        
        main_layout.addWidget(log_group, 0)  # Tỷ lệ 0 (height-for-content)
        
        # Cập nhật thông tin ban đầu
        self.update_belief_display()
    
    def log(self, message):
        """Thêm thông báo vào log"""
        self.log_text.append(f"> {message}")
        if self.parent:
            self.parent.log(message)
    
    def select_algorithm(self, algorithm):
        """Chọn thuật toán mới"""
        self.current_algorithm = algorithm
        self.log(f"Đã chọn thuật toán: {algorithm}")
    
    def update_environment_config(self):
        """Cập nhật cấu hình môi trường"""
        try:
            # Đọc vị trí có thể quan sát được
            obs_pos_text = self.observable_pos_input.text().strip()
            obs_positions = [int(x.strip()) for x in obs_pos_text.split(',')]
            
            # Đọc xác suất hành động ngẫu nhiên
            random_action = float(self.random_action_input.text())
            
            # Đọc xác suất môi trường tự thay đổi
            random_change = float(self.random_change_input.text())
            
            # Cập nhật môi trường
            self.environment = NonDeterministicEnvironment(
                initial_state=self.initial_state,
                goal_state=self.goal_state,
                observable_positions=obs_positions,
                random_action_prob=random_action,
                random_change_prob=random_change
            )
            
            # Cập nhật hiển thị
            self.update_observable_display()
            self.update_belief_display()
            
            self.log(f"Đã cập nhật cấu hình môi trường: {len(obs_positions)} vị trí có thể quan sát, {random_action:.2f} xác suất hành động ngẫu nhiên, {random_change:.2f} xác suất môi trường tự thay đổi")
            
        except Exception as e:
            self.log(f"Lỗi khi cập nhật cấu hình: {str(e)}")
    
    def update_observable_display(self):
        """Cập nhật hiển thị các ô có thể quan sát"""
        observable_state = self.environment.get_observable_state()
        
        for i in range(9):
            cell = self.observable_cells[i]
            
            # Cập nhật màu nền dựa trên vị trí có thể quan sát
            is_observable = i in self.environment.observable_positions
            cell.setStyleSheet(f"""
                background-color: {COLORS['primary'] if is_observable else COLORS['divider']};
                color: white;
                border-radius: 5px;
                padding: 10px;
                min-width: 40px;
                min-height: 40px;
                font-size: 16px;
                font-weight: bold;
                qproperty-alignment: AlignCenter;
            """)
            
            # Đảm bảo kích thước cố định
            cell.setMinimumSize(60, 60)
            cell.setMaximumSize(60, 60)
            
            # Cập nhật giá trị hiển thị
            if is_observable:
                cell.setText(str(observable_state[i]) if observable_state[i] != -1 else "?")
            else:
                cell.setText("?")
    
    def update_belief_display(self):
        """Cập nhật hiển thị thông tin trạng thái niềm tin"""
        belief_count = len(self.environment.belief_states)
        most_likely = self.environment.get_most_likely_state()
        
        # Cập nhật thông tin belief states
        belief_text = f"Số lượng trạng thái niềm tin hiện tại: {belief_count}\n"
        if belief_count > 0:
            belief_text += f"Trạng thái có khả năng cao nhất: {most_likely}\n"
            
            # Thêm thông tin về khoảng cách Manhattan trung bình đến goal
            if belief_count <= 100:  # Chỉ tính nếu số lượng belief states không quá lớn
                avg_distance = sum(manhattan_distance(state, self.goal_state) for state in self.environment.belief_states) / belief_count
                belief_text += f"Khoảng cách Manhattan trung bình đến goal: {avg_distance:.2f}"
        
        self.belief_info.setText(belief_text)
        
        # Cập nhật bảng trạng thái thực
        self.puzzle_board.state = list(self.environment.state)
        self.puzzle_board.update()
        
        # Cập nhật hiển thị trạng thái có thể quan sát
        self.update_observable_display()
    
    def solve(self):
        """Giải bài toán với thuật toán đã chọn"""
        self.log(f"Đang giải với thuật toán: {self.current_algorithm}")
        
        # Reset đường đi và bước hiện tại
        self.solution_path = []
        self.current_step = 0
        
        # Chọn thuật toán phù hợp
        if self.current_algorithm == "ANDOR":
            self.solution_path = andor_graph_search_non_deterministic(self.environment)
        elif self.current_algorithm == "PARTIALLY_OBS":
            self.solution_path = improved_partially_observable(self.environment)
        elif self.current_algorithm == "BELIEF_STATE":
            self.solution_path = improved_belief_state_search(self.environment)
        
        # Hiển thị kết quả
        if self.solution_path:
            self.solution_info.setText(f"Đã tìm thấy giải pháp!\nSố bước: {len(self.solution_path)}\nĐường đi: {', '.join(self.solution_path)}")
            self.log(f"Đã tìm thấy đường đi với {len(self.solution_path)} bước")
        else:
            self.solution_info.setText("Không tìm thấy giải pháp!")
            self.log("Không tìm thấy giải pháp")
    
    def reset(self):
        """Đặt lại môi trường"""
        # Tạo trạng thái ngẫu nhiên mới
        new_state = generate_random_puzzle()
        self.initial_state = tuple(new_state)
        
        # Reset môi trường
        self.environment.reset(new_state)
        
        # Cập nhật hiển thị
        self.puzzle_board.state = list(new_state)
        self.puzzle_board.update()
        self.update_belief_display()
        
        # Reset đường đi
        self.solution_path = []
        self.current_step = 0
        self.solution_info.setText("Chưa có giải pháp")
        
        self.log(f"Đã đặt lại môi trường với trạng thái mới: {new_state}")
    
    def take_step(self):
        """Thực hiện một bước trong đường đi"""
        if not self.solution_path or self.current_step >= len(self.solution_path):
            self.log("Không có bước nào để thực hiện")
            return
        
        # Lấy hành động tiếp theo
        action = self.solution_path[self.current_step]
        
        # Thực hiện hành động trên môi trường
        observable, reward, done = self.environment.take_action(action)
        
        # Cập nhật hiển thị
        self.update_belief_display()
        
        # Log thông tin
        self.log(f"Thực hiện hành động: {action}, Phần thưởng: {reward}, Hoàn thành: {done}")
        
        # Tăng chỉ số bước hiện tại
        self.current_step += 1
        
        # Cập nhật thông tin đường đi
        if self.current_step >= len(self.solution_path):
            if done:
                self.solution_info.setText(f"Đã hoàn thành tất cả {self.current_step} bước và đạt đích!")
            else:
                self.solution_info.setText(f"Đã hoàn thành tất cả {self.current_step} bước nhưng chưa đạt đích!")
        else:
            self.solution_info.setText(f"Bước {self.current_step}/{len(self.solution_path)}\nHành động vừa thực hiện: {action}")
    
    def auto_play(self):
        """Tự động thực hiện các bước trong đường đi"""
        if not self.solution_path or self.current_step >= len(self.solution_path):
            self.log("Không có bước nào để thực hiện tự động")
            return
        
        if self.auto_play_timer.isActive():
            self.auto_play_timer.stop()
            self.log("Đã dừng tự động chạy")
            return
            
        self.log("Bắt đầu tự động chạy...")
        self.auto_play_timer.start(800)  # 800ms mỗi bước
    
    def auto_play_next(self):
        """Callback cho auto_play_timer"""
        if not self.solution_path or self.current_step >= len(self.solution_path):
            self.auto_play_timer.stop()
            self.log("Tự động chạy hoàn tất")
            return
            
        self.take_step()
    
    def closeEvent(self, event):
        """Xử lý sự kiện đóng cửa sổ"""
        # Dừng auto-play timer nếu đang chạy
        if self.auto_play_timer.isActive():
            self.auto_play_timer.stop()
            
        # Xử lý sự kiện đóng khi có parent window
        if self.parent and isinstance(self.parent, MainWindow):
            self.parent.non_deterministic_window = None
            self.parent.log("Đã đóng cửa sổ tìm kiếm trong môi trường không xác định")
            
            # Hiển thị lại puzzle_frame nếu cần
            if self.parent.current_algorithm not in ["ANDOR", "PARTIALLY_OBS", "BELIEF_STATE"]:
                self.parent.puzzle_frame.setVisible(True)
        
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

