#!/usr/bin/env bash

bag_path="/media/XXX/davis240c"

bag_name=(
"boxes_6dof"
"boxes_translation"
"dynamic_6dof"
"dynamic_translation"
"hdr_boxes"
"hdr_poster"
"poster_6dof"
"poster_translation"
"shapes_6dof"
"shapes_translation"
)


project_path="/home/XXXX"
result_path="/media/XXXX"

estimate_file="traj_estimate.txt"
timing_file="traj_timing.txt"

source_string="devel/setup.bash"

dataset_name="davis240c"
test_time="test"
test_seq={0}

# catkin_make
# cd $project_path && catkin_make -j8

bag_count=-1

for i in "${!bag_name[@]}"; do
    let bag_count=bag_count+1

    if [[ ! "${test_seq}" =~ $bag_count ]]; then
        continue
    fi
    echo "run bag ${bag_name[i]}!"

    cd $project_path && source $source_string
    # roslaunch esvo_time_surface stereo_time_surface.launch & pid1=$!
    path_traj="${result_path}/$dataset_name/$test_time/${bag_name[i]}_${estimate_file}"
    path_t="${result_path}/$dataset_name/$test_time/${bag_name[i]}_${timing_file}"
    
    roslaunch ov_msckf subscribe_davis240c.launch dataset:=${bag_name[i]} \
                                            path_est:=${path_traj} \
                                            path_time:=${path_t}

    echo "bag ${bag_name[i]} is done!"
    sleep 3

done
