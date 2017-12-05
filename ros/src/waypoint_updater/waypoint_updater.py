#!/usr/bin/env python
import sys
sys.path.append('/home/student/work/ros/src/styx_msgs/')
import rospy

from geometry_msgs.msg import PoseStamped, TwistStamped
from geometry_msgs.msg import Quaternion

from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import tf
import math
import numpy.polynomial.polynomial as poly
from copy import deepcopy
from scipy.spatial import KDTree

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''


RATE = 10           # update rate: use 10 Hz because positions are received at 10 Hz

LOOKAHEAD_WPS = 50      # Number of waypoints to publish
ACC_WPS_NUM = 4         # Number of waypoints used for acceleration

MAX_VEL_CARLA = 10.0   # default max velocity in kmph for Carla (param server provides speed in km/h)
SAFETY_DISTANCE_FOR_BRAKING = 50                # distance to approach traffic lights 'ready for braking' in meters
SAFETY_SPEED_FOR_BRAKING = MAX_VEL_CARLA / 3.6  # safety speed for approaching traffic lights in m/s

MPS_AS_MPH = 2.23694 # meters per second as miles per hour (as provided in /current_velocity)

PRINT_DEBUG = False  # Print rospy.logwarn for debugging if True


class WaypointUpdater(object):
    def __init__(self):
        rospy.loginfo('Initializing the base model') 
     
        rospy.init_node('waypoint_updater')

        ########### Self parameters  ###############

        self.base_waypoints = None            # base points coming from csv file                
        self.curr_pose = None                 # current pose
        self.final_waypoints =  None          # final waypoints to publish for other nodes
        self.tree = None                      # tree struct for coordinates
        self.curr_velocity = None             # current velocity    
        self.max_velocity = None              # maximum allowed velocity in meters per second        
        self.next_waypoint_index  = None      # Index of the first waypoint in front of the car
        self.traffic_index = None             # waypoint index of next traffic light
        self.traffic_state = None             # state of next traffic light
        
        # Max. velocity from ros parameter server
        max_vel = rospy.get_param("/waypoint_loader/velocity", MAX_VEL_CARLA)
        self.max_velocity = self.kmph2mps(max_vel)       
        
        # Subscribe to topic '/current_pose' to get the current position of the car
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        
        # Subscribe to topic '/current_velocity' to get the current velocity of the car
        rospy.Subscriber('/current_velocity', TwistStamped, callback=self.currvel_cb)

        # Fetch info from '/base_waypoints', which provide a complete list of waypoints the car will be following
        wp = rospy.wait_for_message('/base_waypoints', Lane)
        
        # Set the waypoints only once
        if not self.base_waypoints:
            self.base_waypoints = wp.waypoints  
            rospy.logwarn('Got the base points for waypoint_updater.')        
        
        # Get the x/y coordinates of the base_waypoints
        b_xcor = []
        b_ycor = []
        
        for pt in self.base_waypoints:
            b_xcor.append(pt.pose.pose.position.x)
            b_ycor.append(pt.pose.pose.position.y)
        self.tree = KDTree(zip(b_xcor, b_ycor))
        
        #rospy.logwarn('Got %s base waypoints!',len(b_xcor))
        
        # Subscribe to topic '/traffic_waypoint' to get the locations to stop for red traffic lights
        rospy.Subscriber('/traffic_waypoint', Int32 ,self.traffic_cb)
        rospy.Subscriber('/traffic_waypoint_state', Int32, self.traffic_state_cb);
        
        # TODO: Include if obstacle detection implementation is included later on
        #rospy.Subscriber('/obstacle_waypoint',,self.obstacle_cb)
        
        # Create the publisher to write messages to topic '/final_waypoints'
        # The next waypoints the car has to follow are published
        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        # Run the loop the handle action
        self.loop_n_sleep()


   
    ### Begin: Callback functions for subsribers to ROS topics

    # Callback function to set the current velocity of the car
    # Information provided by ros topic '/current_velocity' in meters per second
    def currvel_cb(self,msg):
        self.curr_velocity = msg.twist.linear.x
    
    # Callback function to set the current position of the car
    # Information provided by ros topic '/current_pose'
    def pose_cb(self, msg):
        self.curr_pose = msg.pose

    # Callback function to get the locations to stop for red traffic lights
    # Information provided by ros topic '/traffic_waypoint'
    def traffic_cb(self, msg):
        self.traffic_index = msg.data
        
    def traffic_state_cb(self, msg):
        self.traffic_state = msg.data

    # Callback function to get the position of obstacles
    # Information provided by ros topic '/obstacle_waypoint'
    # TODO: Subsriber not used at the moment (see __init__)
    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    ### End: Callback functions for subsribers to ROS topics


 
    # Function to get the velocity in x direction (car coordinate system) of a single waypoint
    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    # Function to set the velocity in x direction (car coordinate system) of a single waypoint
    def set_waypoint_velocity(self, waypoint, velocity):
        waypoint.twist.twist.linear.x = velocity


    # Function to get the current location with respect to base points, 
    # and draw next location path, publish this path and then sleep
    def loop_n_sleep(self):
        # Set the update rate
        rate = rospy.Rate(RATE)
        
        # Run as long as ros node is not shut down
        while not rospy.is_shutdown():
            # Check if data is available
            if self.base_waypoints and self.curr_pose:
                #rospy.logwarn('Entering the publisher now')
                
                closest_wp_idx = self.find_next_waypoint_idx(
                                    self.base_waypoints,
                                    self.tree,
                                    self.curr_pose.position.x,
                                    self.curr_pose.position.y,
                                    self.curr_pose.orientation.w)
                
                ## Set the index of the closest waypoint in front of the car
                self.next_waypoint_index = closest_wp_idx
                #rospy.logwarn('Closest index is %s', self.next_waypoint_index )                
                
                # Get the values for velocities for the upcoming waypoints
                value_waypoint_velocities = self.get_waypoint_velocities()

                # Publish the next waypoints the car should follow
                self.publish(self.next_waypoint_index, value_waypoint_velocities)
        
            rate.sleep()

    # Publish the next waypoints the car should follow
    def publish(self, idx, waypoint_velocities):

        # Create Lane object and set timestamp
        final_waypoints_msg = Lane()
        final_waypoints_msg.header.stamp = rospy.Time.now()
    
        # Update waypoints and set their velocities. 
        self.final_waypoints = deepcopy(self.base_waypoints[idx: idx + LOOKAHEAD_WPS])

        # for pt in self.final_waypoints:
        #   rospy.logwarn('Next point is %s %s ',pt.pose.pose.position.x, pt.pose.pose.position.y)
        
        #rospy.logwarn(waypoint_velocities[0]*2.23694)
        
        for i in range(LOOKAHEAD_WPS):
            self.set_waypoint_velocity(self.final_waypoints[i], waypoint_velocities[i])      
        
        # Set waypoints in waypoint message
        final_waypoints_msg.waypoints = self.final_waypoints
        
        # Publish the waypoints the car should follow to ros topic '/final_waypoints'
        self.final_waypoints_pub.publish(final_waypoints_msg)

    
    # Decelerate evenly and stop at the stop line
    def brake_till_traffic_waypoint(self):
        # Array for waypoint velocities
        waypoint_velocities = []
        
        # Calculate the difference between current speed and final target speed
        diff_index = self.traffic_index - self.next_waypoint_index
        rospy.logwarn('brake_till_traffic_waypoint: diff_index=%i', diff_index)

        # Prevent negative index
        if diff_index < 0:
            diff_index = 0

        # Calculate how much the velocity should be reduced per waypoint
        diff_velocity = self.curr_velocity / (diff_index + 1)

        new_velocity = self.curr_velocity

        for i in range(LOOKAHEAD_WPS):
            # Before traffic sign
            if i < diff_index:
                new_velocity -= diff_velocity
                # If target velocity is really small -> set to zero
                if new_velocity < 0.1:
                    new_velocity = 0
                
                waypoint_velocities.append(new_velocity * 0.7) # TODO: find better solution
            # After traffic sign -> set all to zero
            else:
                waypoint_velocities.append(0)

        if PRINT_DEBUG:
            rospy.logwarn('Brake!! Current v: %.2f mph, target v: %.2f:%.2f:%.2f:%.2f:%.2f (mph).', 
                          self.curr_velocity * 2.23694, 
                          waypoint_velocities[0] * 2.23694, 
                          waypoint_velocities[1] * 2.23694, 
                          waypoint_velocities[2] * 2.23694, 
                          waypoint_velocities[3] * 2.23694, 
                          waypoint_velocities[4] * 2.23694)

        return waypoint_velocities   
    
    def approach_traffic_light(self):
        ''' 
        reduce speed if necessary to be prepared to stop in case traffic 
        light turns red|yellow
        '''
        diff_index = self.traffic_index - self.next_waypoint_index
        # rospy.logwarn('approach_traffic_light: diff_index=%i', diff_index)
        
        # Prevent negative index
        if diff_index <= 0:
            diff_index = 1
            
        delta_v = SAFETY_SPEED_FOR_BRAKING - self.curr_velocity
        step_v = delta_v / diff_index 
        
        # Array for waypoint velocities
        waypoint_velocities = []
        
        v = self.curr_velocity
        for _ in range(LOOKAHEAD_WPS):
            if v > SAFETY_SPEED_FOR_BRAKING:
                v -= step_v
            
            if v < SAFETY_SPEED_FOR_BRAKING:
                v = SAFETY_SPEED_FOR_BRAKING
                
            waypoint_velocities.append(v)
        
        return waypoint_velocities
        
    
    # Set the target velocity for all waypoints within LOOKAHEAD_WPS to zero
    def set_velocity_to_zero(self):
        # Array for waypoint velocities
        waypoint_velocities = []

        for i in range(LOOKAHEAD_WPS):
                waypoint_velocities.append(0)

        if PRINT_DEBUG:
            rospy.logwarn('Speed Zero!! Current v: %.2f mph, target v: %.2f:%.2f:%.2f:%.2f:%.2f (mph).', self.curr_velocity * 2.23694, waypoint_velocities[0] * 2.23694, waypoint_velocities[1] * 2.23694, waypoint_velocities[2] * 2.23694, waypoint_velocities[3] * 2.23694, waypoint_velocities[4] * 2.23694)

        return waypoint_velocities

    # Accelerate evenly up to the allowed speed limit
    def speed_up_to_max(self):
        # Array for waypoint velocities
        waypoint_velocities = []

        # Check if current velocity is smaller than max_velocity
        if self.curr_velocity <= self.max_velocity: 
            # Calculate how much the velocity should be raised per waypoint
            diff_velocity = (self.max_velocity - self.curr_velocity) / ACC_WPS_NUM
        # If too fast -> reduce speed
        else:
            diff_velocity = self.curr_velocity - self.max_velocity

        new_velocity = self.curr_velocity

        for i in range(LOOKAHEAD_WPS):
            # Before reaching max_velocity
            if i < ACC_WPS_NUM:
                new_velocity += diff_velocity
                
                if new_velocity > self.max_velocity:
                    new_velocity = self.max_velocity

                waypoint_velocities.append(new_velocity)
            # After reaching the speed limit
            else:
                waypoint_velocities.append(self.max_velocity)

        # if PRINT_DEBUG:
        #     rospy.logwarn('Speed up!! Current v: %.2f mph, target v: %.2f:%.2f:%.2f:%.2f:%.2f (mph).', self.curr_velocity * 2.23694, waypoint_velocities[0] * 2.23694, waypoint_velocities[1] * 2.23694, waypoint_velocities[2] * 2.23694, waypoint_velocities[3] * 2.23694, waypoint_velocities[4] * 2.23694)

        return waypoint_velocities

    # Move slowly to the stop line
    def move_slowly_to_waypoint(self):
        # Array for waypoint velocities
        waypoint_velocities = []

        # Calculate the difference between current speed and final target speed
        diff_index = self.traffic_index - self.next_waypoint_index

        for i in range(LOOKAHEAD_WPS):
            # Before traffic sign idx -1 (Stop one waypoint before stop line SAFETY)
            if i < diff_index - 1: 
                waypoint_velocities.append(1)
            # After traffic sign
            else:
                waypoint_velocities.append(0)

        if PRINT_DEBUG:
            rospy.logwarn('Move slowly:  Current v: %.2f mph, target v: %.2f:%.2f:%.2f:%.2f:%.2f (mph).', 
                          self.curr_velocity * 2.23694, 
                          waypoint_velocities[0] * 2.23694, 
                          waypoint_velocities[1] * 2.23694, 
                          waypoint_velocities[2] * 2.23694, 
                          waypoint_velocities[3] * 2.23694, 
                          waypoint_velocities[4] * 2.23694)

        return waypoint_velocities


    # Get the target velocities for all waypoint within LOOKAHEAD_WPS
    def get_waypoint_velocities(self):
        # Array for waypoint velocities
        waypoint_velocities = []
        
        # Check if necessary data is available
        if self.traffic_index is None:
            rospy.logwarn('traffic_index is None')
            waypoint_velocities = self.set_velocity_to_zero()
        elif self.base_waypoints is None:
            rospy.logwarn('base_waypoints is None')
            waypoint_velocities = self.set_velocity_to_zero()
        elif self.next_waypoint_index is None:
            rospy.logwarn('next_waypoint_index is None')
            waypoint_velocities = self.set_velocity_to_zero()
        elif self.traffic_index is None:
            rospy.logwarn('traffic_index is None')
            waypoint_velocities = self.set_velocity_to_zero()
        elif self.traffic_state is None:
            rospy.logwarn('traffic_state is None')
            waypoint_velocities = self.set_velocity_to_zero()
        else:
            distance_to_traffic_waypoint = self.distance_between_waypoints(
                self.base_waypoints, 
                self.next_waypoint_index, 
                self.traffic_index)

            if PRINT_DEBUG:
                rospy.logwarn('Distance between next waypoint %d and traffic_waypoint %d: %f m.', 
                              self.next_waypoint_index, 
                              self.traffic_index, 
                              distance_to_traffic_waypoint)
            
            if (distance_to_traffic_waypoint < SAFETY_DISTANCE_FOR_BRAKING):
                # slow down to safety speed to prepare for braking OR
                # brake for red/yellow traffic light
            
                if (self.traffic_state == 0 or self.traffic_state == 1):
                    # A red or yellow traffic light is in front of the car
                    if self.curr_velocity < 2:
                        if PRINT_DEBUG:
                            rospy.logwarn('Move slowly to stopping point!')
                        waypoint_velocities = self.move_slowly_to_waypoint()
                    else:
                        rospy.logwarn('Brake for traffic light!')
                        waypoint_velocities = self.brake_till_traffic_waypoint()
                else:
                    # approach traffic light with safety speed
                    waypoint_velocities = self.approach_traffic_light()
            else:
                # No traffic light within safety distance -> speed up to speed limit
                # if PRINT_DEBUG:
                #    rospy.logwarn('No red/yellow traffic light within %d m -> Speed up.', SAFETY_DISTANCE_FOR_BRAKING)
                waypoint_velocities = self.speed_up_to_max()  
            
        return waypoint_velocities

    # Get the distance between two waypoint out of a list of waypoints
    def distance_between_waypoints(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


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

    def kmph2mps(self, velocity_kmph):
        return velocity_kmph / 3.6


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
