sh set_pos.sh 10 0 10
sleep 15
sh add_cali.sh
sleep 5
sh set_pos.sh 15 0 20
sleep 15
sh add_cali.sh
sleep 5
sh set_pos.sh -15 0 30
sleep 15
sh add_cali.sh
sleep 5
sh calc_cali.sh
sleep 5
sh reloc.sh /home/ps/catkin_ws/src/hloc/data/town/query_test/img48.jpg