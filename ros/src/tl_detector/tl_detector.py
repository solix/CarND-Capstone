#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import math

STATE_COUNT_THRESHOLD = 3
PRINT_DEBUG = True              # Print rospy.logwarn for debugging if True
USE_GROUND_TRUTH_STATE = False   # True if traffic light state should be taken from ground truth data

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.base_waypoints = None
        self.curr_pose = None
        self.camera_image = None
        self.tree = None
        self.lights = []
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.datasize = 0
        self.ignore_state = False

        # Subscribe to topic '/current_pose' to get the current position of the car
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        
        # Subscribe to topic '/image_color' to get the image from the cars front camera
        rospy.Subscriber('/image_color', Image, self.image_cb)

        # Subscribe to topic '/vehicle/traffic_lights' to get the location and state of the traffic lights
        # IMPORTANT: The state will not be available in real life testing
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_lights_cb)
        
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # Fetch info from '/base_waypoints', which provide a complete list of waypoints the car will be following
        wp = rospy.wait_for_message('/base_waypoints', Lane)
        
        # Set the waypoints only once
        if not self.base_waypoints:
            self.base_waypoints = wp.waypoints
            self.datasize = len(self.base_waypoints)
            if PRINT_DEBUG:
                rospy.logwarn('Got the base points for tl_detector.')        
        
        # Get the x/y coordinates of the base_waypoints
        b_xcor = []
        b_ycor = []
        
        for pt in self.base_waypoints:
            b_xcor.append(pt.pose.pose.position.x)
            b_ycor.append(pt.pose.pose.position.y)
        self.tree = KDTree(zip(b_xcor, b_ycor))


        # Create the publisher to write messages to topic '/traffic_waypoint'
        # The index of the waypoint which is closest to the next red traffic light has to be published
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        # Create the publisher to write messages to topic '/traffic_waypoint_state'
        # The state of the closest red traffic light has to be published
        self.upcoming_red_light_state_pub = rospy.Publisher('/traffic_waypoint_state', Int32, queue_size=1)

        # Block until shutdown -> Tasks are handled with callbacks
        rospy.spin()


    ### Begin: Callback functions for subsribers to ROS topics

    # Callback function to set the current position of the car
    # Information provided by ros topic '/current_pose'
    def pose_cb(self, msg):
        self.curr_pose = msg.pose

    # Callback function to get the status of the traffic lights
    # Information provided by ros topic '/vehicle/traffic_lights'
    def traffic_lights_cb(self, msg):
        self.lights = msg.lights

    # Callback function to get the status of the traffic lights
    # Information provided by ros topic '/image_color'
    # Identifies red lights in the incoming camera image and publishes the index
    # of the waypoint closest to the red light's stop line to /traffic_waypoint
    # Args: msg (Image): image from car-mounted camera
    def image_cb(self, msg):
        # Set Parameters
        self.has_image = True
        self.camera_image = msg
        
        # Check if the base_waypoints have been loaded successfully
        if not self.base_waypoints: 
            #rospy.logwarn('No base_waypoints \n\n')
            pass

        light_wp, state = self.process_traffic_lights()

        # Publish upcoming red lights at camera frequency.
        # Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        # of times till we start using it. Otherwise the previous stable state is used.
        if self.ignore_state:
            self.upcoming_red_light_pub.publish(Int32(light_wp))
            self.upcoming_red_light_state_pub.publish(Int32(self.state))
        else:
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
                self.upcoming_red_light_state_pub.publish(Int32(self.state))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                self.upcoming_red_light_state_pub.publish(Int32(self.last_state))
            
            self.state_count += 1

    ### End: Callback functions for subsribers to ROS topics

    
    # Determines the current color of the traffic light
    # Args: light (TrafficLight): light to classify
    # Returns: int: ID of traffic light color (specified in styx_msgs/TrafficLight)
    def get_light_state(self, light):
        # Check if an image is available
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        # Get current image
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get the classification of the image
        return self.light_classifier.get_classification(cv_image)

    
    # Finds closest visible traffic light, if one exists, and determines its location and color
    # Returns:
    #       int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
    #       int: ID of traffic light color (specified in styx_msgs/TrafficLight)
    def process_traffic_lights(self):
        light = None

        if(self.curr_pose):
            car_position_idx = self.find_next_waypoint_idx(
                                self.base_waypoints,
                                self.tree,
                                self.curr_pose.position.x,
                                self.curr_pose.position.y,
                                self.curr_pose.orientation.w)

            # List of positions that correspond to the line to stop in front of for a given intersection
            stop_line_positions = self.config['stop_line_positions']

            # Get the index of the waypoint (wp) which is closest to the next stop line
            # Get the index of the corresponding traffic light for the lights array
            traffic_light_wp_idx, traffic_light_idx = self.find_next_traffic_light_idx(stop_line_positions, car_position_idx)

            # Get the distance between the actual car position and the next stop line
            distance_to_traffic_waypoint = self.distance_between_waypoints(self.base_waypoints, car_position_idx, traffic_light_wp_idx)

            # Get next traffic light
            light = self.lights[traffic_light_idx]

            # Print some debug information if needed
            if PRINT_DEBUG:
                rospy.logwarn('Car idx: %d, wp stop line idx: %d, Light nr: %d, Light state: %d, distance %.2f m', car_position_idx, traffic_light_wp_idx, traffic_light_idx, light.state, distance_to_traffic_waypoint)

        # Traffic light available
        # IMPORTANT: Decide if state should be taken from ground truth or camera
        if light:
            self.ignore_state = False

            if USE_GROUND_TRUTH_STATE:
                state = light.state
                if PRINT_DEBUG:
                    rospy.logwarn("GT: %s", state)
            elif distance_to_traffic_waypoint > 100:
                state = TrafficLight.UNKNOWN
                self.ignore_state = True
                if PRINT_DEBUG:
                    rospy.logwarn("Ignore Classifier!")
            else:
                state = self.get_light_state(light)
                if PRINT_DEBUG:
                    rospy.logwarn("CL: %s", state)

            return traffic_light_wp_idx, state
        
        # No detectable traffic light found
        return -1, TrafficLight.UNKNOWN


    # Finds the index of closest traffic light (in front) 
    def find_next_traffic_light_idx(self, stop_line_positions, car_position_idx):
        found_idx = 0
        stop_line_idx = float('inf')

        # get index of the closest stop line    
        for i in range (len(stop_line_positions)):
            pt = PoseStamped()
            pt.pose.position.x = stop_line_positions[i][0]
            pt.pose.position.y = stop_line_positions[i][1]
            pt.pose.position.z = 0
            
            # Get the index of the closest stop line
            idx = self.find_next_waypoint_idx(
                                        self.base_waypoints,
                                        self.tree,
                                        pt.pose.position.x,
                                        pt.pose.position.y,
                                        self.curr_pose.orientation.w)
           
            # Check if idx is ahead of us and is closest
            if (idx >= car_position_idx) and (idx <= stop_line_idx):
                stop_line_idx = idx         # if yes, this is our closest stop line
                found_idx = i
                break

        if stop_line_idx == float('inf') or stop_line_idx >= self.datasize:
            return -1, found_idx
        else:
            return stop_line_idx, found_idx

    # Find the index of closest waypoint (in front) in world waypoints for given position 
    # @param waypoints: received waypoints from waypoint_loader
    def find_next_waypoint_idx(self, waypoints, waypoint_tree, egoX, egoY, egoYaw):
        wp_idx = 0
        
        if not waypoint_tree is None:
            # Using search tree to find next waypoint idx
            _, wp_idx = waypoint_tree.query((egoX, egoY))
            
            closest_wp = waypoints[wp_idx]
            # Convert to cars local coordinate system
            localX, _, _ = self.world_to_local(
                egoX, egoY, egoYaw,
                closest_wp.pose.pose.position.x,
                closest_wp.pose.pose.position.y)
            
            # x axis points in direction of ego vehicle
            # Waypoint in front
            if localX >= 0:
                return wp_idx
            
            # Take first waypoint in front
            wp_idx = (wp_idx + 1) % len(waypoints)

        return wp_idx


    # Transform given world coordinate into local coordinate:
    # positive x points forward, positive y points left
    # @return local x, y and relative angle to world point
    def world_to_local(self, egoX, egoY, egoYaw, worldX, worldY):
        # Translate
        dX = worldX - egoX
        dY = worldY - egoY
        
        # Rotate
        localX = math.cos(-egoYaw) * dX + math.sin(-egoYaw) * dY
        localY = -math.sin(-egoYaw) * dY + math.cos(-egoYaw) * dY
        
        # Calculate relative angle
        relAngle = math.atan2(localY, localX)
        
        return localX, localY, relAngle

    # Get the distance between two waypoint out of a list of waypoints
    def distance_between_waypoints(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
