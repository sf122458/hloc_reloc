rostopic pub -1 /set_pos geometry_msgs/PoseStamped "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
pose:
  position:
    x: $1
    y: $2
    z: $3
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 0.0"