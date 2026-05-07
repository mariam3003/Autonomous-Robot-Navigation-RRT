import cv2
import numpy as np
import random
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import heapq

WIDTH = 800
HEIGHT = 600
NUM_POINTS = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
BLUE = (255, 0, 0)
LIGHT_BLUE = (173, 216, 230)
PURPLE = (128, 0, 128)

GOAL_X, GOAL_Y = 182, 560.5
START_X, START_Y = 100, 100

robot_radius = 0.269
pixels = 50
dilation = int(robot_radius * pixels)


def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class RobotTrackerNode(Node):
    def __init__(self):
        super().__init__('robot_tracker')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self._publisher = self.create_publisher(Twist, "/cmd_vel", 1)
        self.maze = self.loading_maze_visualization('/home/common/Downloads/40px_new.png', dilation)
        scale_x = self.resizing_maze(self.maze, WIDTH, HEIGHT)
        scale_y = self.resizing_maze(self.maze, WIDTH, HEIGHT)
        self.world = self.creating_maze_imagery(self.maze, scale_x, scale_y)
        self.draw_grid()
        self.points = self.distribute_random_points(self.world, self.maze, NUM_POINTS, scale_x, scale_y)
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.tree_nodes = [(START_X, START_Y)]
        self.tree_edges = []
        self.tree_graph = {}
        self.start_connected = False
        self.goal_connected = False
        self.start_node = None
        self.goal_node = None
        self.waypoints = []
        self._goal_found_to_drive = False
        self.current_goal_index = 0
        self.place_goal_on_map()
        self.get_logger().info(f'{self.get_name()} node has started.')
        self.robot_position_set = False

    def loading_maze_visualization(self, image_path, dilation):
        maze_imagery = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if maze_imagery is None:
            raise ValueError("Image is invalid or not found.")
        maze = np.where(maze_imagery == 0, 1, 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        maze = cv2.dilate(maze.astype(np.uint8), kernel, iterations=1)
        return maze

    def resizing_maze(self, maze, world_width, world_height):
        maze_height, maze_width = maze.shape
        scale_x = world_width / maze_width
        scale_y = world_height / maze_height
        return scale_x, scale_y

    def creating_maze_imagery(self, maze, scale_x, scale_y):
        height, width = maze.shape
        world = np.full((HEIGHT, WIDTH, 3), WHITE, dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                top_left = (int(x * scale_x), int(y * scale_y))
                bottom_right = (int((x + 1) * scale_x), int((y + 1) * scale_y))
                if maze[y, x] == 1:
                    cv2.rectangle(world, top_left, bottom_right, BLACK, thickness=-1)
                elif maze[y, x] == 0:
                    cv2.rectangle(world, top_left, bottom_right, WHITE, thickness=-1)
        return world

    def distribute_random_points(self, world, maze, num_points, scale_x, scale_y):
        points = []
        height, width = maze.shape
        num_inside = num_points
        for _ in range(num_inside):
            while True:
                x = random.uniform(0, WIDTH - 1)
                y = random.uniform(0, HEIGHT - 1)
                if not self.is_point_in_obstacle(maze, (x, y), scale_x, scale_y):
                    color = GREEN
                    points.append(((x, y), color))
                    cv2.circle(world, (int(x), int(y)), radius=3, color=color, thickness=-1)
                    break
        return points

    def point_to_cell(self, point, scale_x, scale_y):
        return (int(point[0] // scale_x), int(point[1] // scale_y))

    def is_point_in_obstacle(self, maze, point, scale_x, scale_y):
        cell_x, cell_y = self.point_to_cell(point, scale_x, scale_y)
        return maze[cell_y, cell_x] == 1

    def finding_nearest(self, tree, point):
        return min(tree, key=lambda node: math.hypot(node[0] - point[0], node[1] - point[1]))

    def can_connect(self, world, maze, start, end, scale_x, scale_y):
        x1, y1 = start
        x2, y2 = end
        num_points = int(math.hypot(x2 - x1, y2 - y1))
        for i in range(num_points):
            x = int(x1 + i * (x2 - x1) / num_points)
            y = int(y1 + i * (y2 - y1) / num_points)
            if self.is_point_in_obstacle(maze, (x, y), scale_x, scale_y):
                return False
        return True

    def draw_grid(self, grid_size=42):
        for x in range(0, WIDTH, grid_size):
            cv2.line(self.world, (x, 0), (x, HEIGHT), (220, 220, 220), 1)
        for y in range(0, HEIGHT, grid_size):
            cv2.line(self.world, (0, y), (WIDTH, y), (220, 220, 220), 1)

    def _short_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _computing_speed(self, diff, max_speed, min_speed, gain):
        """ Compute the speed based on the difference (error), with speed limits """
        speed = abs(diff) * gain
        speed = min(max_speed, max(min_speed, speed))
        return math.copysign(speed, diff)

    def _drive_to_goal(self, goal_x, goal_y, goal_theta, vel_gain=0.5, angular_gain=2.5,
                       max_vel=0.2, max_angular_vel=0.8, max_pos_err=0.15, max_angle_err=0.12):
        pose = self.r_pose
        cur_x = pose.position.x
        cur_y = pose.position.y
        o = pose.orientation
        roll, pitch, yaw = euler_from_quaternion(o)
        cur_t = yaw

        desired_angle = math.atan2(goal_y - cur_y, goal_x - cur_x)
        angle_diff = self._short_angle(desired_angle - cur_t)
        x_diff = goal_x - cur_x
        y_diff = goal_y - cur_y
        dist = math.sqrt(x_diff * x_diff + y_diff * y_diff)

        twist = Twist()
        if abs(angle_diff) > max_angle_err:
            angular_vel = self._computing_speed(angle_diff, max_angular_vel, -max_angular_vel, angular_gain)
            twist.angular.z = angular_vel
        else:
            if dist > max_pos_err:
                linear_vel = self._computing_speed(dist, max_vel, -max_vel, vel_gain)
                twist.linear.x = linear_vel
                self.get_logger().info(f"{dist}")
            else:
                self.get_logger().info(f"Reached goal")
                return True
        self._publisher.publish(twist)
        return False

    def odom_callback(self, msg: Odometry, vel_gain=5.0, max_vel=0.2,
                      max_pos_err=0.05, angular_gain=3.0, angular_vel=1.0, angular_err=0.1):
        pose = msg.pose.pose
        self.r_pose = pose
        x, y = pose.position.x, pose.position.y
        o = pose.orientation
        roll, pitchc, yaw = euler_from_quaternion(o)
        cur_t = yaw
        self.robot_angle = cur_t
        self.robot_gazebo_pos = x, y
        self.robot_position = self.translating_to_original((x, y), (10, 8))
        self.robot_position_set = True
        self.update_robot_on_map()

        if self._goal_found_to_drive:
            if self.waypoints:
                goal_x, goal_y = self.translating_point(self.waypoints[self.current_goal_index])
                goal_reached = self._drive_to_goal(goal_x, goal_y, goal_theta=0)
                self.get_logger().info(f"Going to goal {goal_x},{goal_y} current goal index={self.current_goal_index}")
                if goal_reached:
                    self.current_goal_index = self.current_goal_index + 1
                    print("here")
                    if self.current_goal_index > len(self.waypoints):
                        self.waypoints = []
                        self._goal_found_to_drive = False

    def translating_to_original(self, point, origin):
        x, y = point
        ox, oy = origin
        calc_x = (x + ox) * 42
        calc_y = (oy - y) * 42
        return (calc_x, calc_y)

    def translating_point(self, point, origin=(10 * 42, 8 * 42)):
        x, y = point
        ox, oy = origin
        calc_x = (x - ox)
        scalecalc_x = calc_x / 42
        calc_y = (oy - y)
        scalecalc_y = calc_y / 42
        return (scalecalc_x, scalecalc_y)

    def calculating_angle(self, p1, p2):
        return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

    def place_goal_on_map(self):
        goal_position = (GOAL_X, GOAL_Y)
        cv2.circle(self.world, (int(goal_position[0]), int(goal_position[1])), radius=10, color=PURPLE, thickness=-1)

    def update_robot_on_map(self):
        if self.robot_position_set:
            world_copy = self.world.copy()
            cv2.circle(world_copy, (int(self.robot_position[0]), int(self.robot_position[1])), radius=5, color=RED, thickness=-1)
            cv2.imshow('RRT', world_copy)
            cv2.waitKey(1)

    def growing_rrt(self):
        if not self.robot_position_set:
            return
        start = self.robot_position
        self.get_logger().info("RRT Growth started...")
        self.tree_nodes = [start]
        self.tree_edges = []
        self.tree_graph = {}

        for point, _ in self.points:
            closest_node = self.finding_nearest(self.tree_nodes, point)
            if self.can_connect(self.world, self.maze, closest_node, point, self.scale_x, self.scale_y):
                self.tree_nodes.append(point)
                self.tree_edges.append((closest_node, point))
                if closest_node not in self.tree_graph:
                    self.tree_graph[closest_node] = []
                if point not in self.tree_graph:
                    self.tree_graph[point] = []
                self.tree_graph[closest_node].append(point)
                self.tree_graph[point].append(closest_node)
                cv2.line(self.world, (int(closest_node[0]), int(closest_node[1])),
                         (int(point[0]), int(point[1])), ORANGE, 1)
                self.update_robot_on_map()

    def connect_all(self):
        if not self.robot_position_set:
            return
        start = self.robot_position
        goal = (GOAL_X, GOAL_Y)

        for node in self.tree_nodes:
            if self.can_connect(self.world, self.maze, start, node, self.scale_x, self.scale_y):
                self.tree_nodes.append(start)
                self.tree_edges.append((start, node))
                if start not in self.tree_graph:
                    self.tree_graph[start] = []
                self.tree_graph[start].append(node)
                self.tree_graph[node].append(start)
                cv2.line(self.world, (int(start[0]), int(start[1])), (int(node[0]), int(node[1])), ORANGE, 2)
                break

        for node in self.tree_nodes:
            if self.can_connect(self.world, self.maze, goal, node, self.scale_x, self.scale_y):
                self.tree_nodes.append(goal)
                self.tree_edges.append((goal, node))
                if goal not in self.tree_graph:
                    self.tree_graph[goal] = []
                self.tree_graph[goal].append(node)
                self.tree_graph[node].append(goal)
                cv2.line(self.world, (int(goal[0]), int(goal[1])), (int(node[0]), int(node[1])), ORANGE, 2)
                break

        cv2.imshow('RRT', self.world)
        cv2.waitKey(10)
        self.shortest_path()

    def shortest_path(self):
        if not self.robot_position_set:
            return
        start = self.robot_position
        goal = (GOAL_X, GOAL_Y)

        pq = [(0, start)]
        distances = {start: 0}
        previous = {start: None}
        check = set()

        while pq:
            current_distance, current_node = heapq.heappop(pq)
            if current_node == goal:
                break
            if current_node in check:
                continue
            check.add(current_node)

            for neighbor in self.tree_graph.get(current_node, []):
                distance = math.hypot(current_node[0] - neighbor[0], current_node[1] - neighbor[1])
                new_path = current_distance + distance
                if new_path < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_path
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (new_path, neighbor))

        if goal not in previous:
            self.get_logger().info("Path to goal not found.")
            return

        path = self.rebuilding_path(previous, start, goal)
        self.waypoints = path

        for i in range(len(path) - 1):
            self.get_logger().info(f'from ({path[i][0]},{path[i][1]}) to ({path[i+1][0]},{path[i+1][1]})')
            self._goal_found_to_drive = True
            cv2.line(self.world, (int(path[i][0]), int(path[i][1])),
                     (int(path[i + 1][0]), int(path[i + 1][1])), BLUE, 2)

        cv2.imshow('RRT', self.world)
        cv2.waitKey(10)

    def rebuilding_path(self, previous, start, goal):
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = previous[current]
        path.append(start)
        path.reverse()
        return path


def main(args=None):
    rclpy.init(args=args)
    node = RobotTrackerNode()
    while not node.robot_position_set:
        rclpy.spin_once(node)
    node.growing_rrt()
    node.connect_all()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
