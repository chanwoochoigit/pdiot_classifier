import glob
import pandas as pd
import itertools

path_2018 = 'pdiot-data/2018'
path_2019 = 'pdiot-data/2019'
path_2020 = 'pdiot-data/2020'

files_2018 = glob.glob(path_2018+'/*.csv')
files_2019 = glob.glob(path_2019+'/*.csv')
files_2020 = glob.glob(path_2020+'/*/*.csv')

all_files = [x for x in files_2018+files_2019+files_2020]

#initialisation all classes
walk_chest_left = []
walk_chest_right = []
walk_pocket_left = []
walk_pocket_right = []
walk_wrist_left = []
walk_wrist_right = []

climbing_chest_left = []
climbing_chest_right = []
climbing_pocket_left = []
climbing_pocket_right = []
climbing_wrist_left = []
climbing_wrist_right = []

descending_chest_left = []
descending_chest_right = []
descending_pocket_left = []
descending_pocket_right = []
descending_wrist_left = []
descending_wrist_right = []

deskwork_chest_left = []
deskwork_chest_right = []
deskwork_pocket_left = []
deskwork_pocket_right = []
deskwork_wrist_left = []
deskwork_wrist_right = []

lyingLeft_chest_left = []
lyingLeft_chest_right = []
lyingLeft_pocket_left = []
lyingLeft_pocket_right = []
lyingLeft_wrist_left = []
lyingLeft_wrist_right = []

lyingRight_chest_left = []
lyingRight_chest_right = []
lyingRight_pocket_left = []
lyingRight_pocket_right = []
lyingRight_wrist_left = []
lyingRight_wrist_right = []

lyingBack_chest_left = []
lyingBack_chest_right = []
lyingBack_pocket_left = []
lyingBack_pocket_right = []
lyingBack_wrist_left = []
lyingBack_wrist_right = []

lyingStomach_chest_left = []
lyingStomach_chest_right = []
lyingStomach_pocket_left = []
lyingStomach_pocket_right = []
lyingStomach_wrist_left = []
lyingStomach_wrist_right = []

run_chest_left = []
run_chest_right = []
run_pocket_left = []
run_pocket_right = []
run_wrist_left = []
run_wrist_right = []

sitForward_chest_left = []
sitForward_chest_right = []
sitForward_pocket_left = []
sitForward_pocket_right = []
sitForward_wrist_left = []
sitForward_wrist_right = []

sitBackward_chest_left = []
sitBackward_chest_right = []
sitBackward_pocket_left = []
sitBackward_pocket_right = []
sitBackward_wrist_left = []
sitBackward_wrist_right = []

sitStand_chest_left = []
sitStand_chest_right = []
sitStand_pocket_left = []
sitStand_pocket_right = []
sitStand_wrist_left = []
sitStand_wrist_right = []

for file in all_files:
    if 'chest_left' in file.lower():
        if 'walk' in file.lower():
            walk_chest_left.append(file)
        elif 'climbing' in file.lower():
            climbing_chest_left.append(file)
        elif 'descending' in file.lower():
            descending_chest_left.append(file)
        elif 'desk' in file.lower():
            deskwork_chest_left.append(file)
        elif 'lying down left' in file.lower():
            lyingLeft_chest_left.append(file)
        elif 'lying down right' in file.lower():
            lyingRight_chest_left.append(file)
        elif 'lying' in file.lower() and 'stomach' in file.lower():
            lyingStomach_chest_left.append(file)
        elif 'lying' in file.lower() and 'back' in file.lower():
            lyingBack_chest_left.append(file)
        elif 'run' in file.lower():
            run_chest_left.append(file)
        elif 'sit' in file.lower() and 'forward' in file.lower():
            sitForward_chest_left.append(file)
        elif 'sit' in file.lower() and 'backward' in file.lower():
            sitBackward_chest_left.append(file)
        elif 'sit' in file.lower() and 'forward' not in file.lower() and 'backward' not in file.lower():
            sitStand_chest_left.append(file)
    elif 'chest_right' in file.lower():
        if 'walk' in file.lower():
            walk_chest_right.append(file)
        elif 'climbing' in file.lower():
            climbing_chest_right.append(file)
        elif 'descending' in file.lower():
            descending_chest_right.append(file)
        elif 'desk' in file.lower():
            deskwork_chest_right.append(file)
        elif 'lying down left' in file.lower():
            lyingLeft_chest_right.append(file)
        elif 'lying down right' in file.lower():
            lyingRight_chest_right.append(file)
        elif 'lying' in file.lower() and 'stomach' in file.lower():
            lyingStomach_chest_right.append(file)
        elif 'lying' in file.lower() and 'back' in file.lower():
            lyingBack_chest_right.append(file)
        elif 'run' in file.lower():
            run_chest_right.append(file)
        elif 'sit' in file.lower() and 'forward' in file.lower():
            sitForward_chest_right.append(file)
        elif 'sit' in file.lower() and 'backward' in file.lower():
            sitBackward_chest_right.append(file)
        elif 'sit' in file.lower() and 'forward' not in file.lower() and 'backward' not in file.lower():
            sitStand_chest_right.append(file)
    elif 'pocket' in file.lower() and '_left_' in file.lower():
        if 'walk' in file.lower():
            walk_pocket_left.append(file)
        elif 'climbing' in file.lower():
            climbing_pocket_left.append(file)
        elif 'descending' in file.lower():
            descending_pocket_left.append(file)
        elif 'desk' in file.lower():
            deskwork_pocket_left.append(file)
        elif 'lying down left' in file.lower():
            lyingLeft_pocket_left.append(file)
        elif 'lying down right' in file.lower():
            lyingRight_pocket_left.append(file)
        elif 'lying' in file.lower() and 'stomach' in file.lower():
            lyingStomach_pocket_left.append(file)
        elif 'lying' in file.lower() and 'back' in file.lower():
            lyingBack_pocket_left.append(file)
        elif 'run' in file.lower():
            run_pocket_left.append(file)
        elif 'sit' in file.lower() and 'forward' in file.lower():
            sitForward_pocket_left.append(file)
        elif 'sit' in file.lower() and 'backward' in file.lower():
            sitBackward_pocket_left.append(file)
        elif 'sit' in file.lower() and 'forward' not in file.lower() and 'backward' not in file.lower():
            sitStand_pocket_left.append(file)
    elif 'pocket' in file.lower() and '_right_' in file.lower():
        if 'walk' in file.lower():
            walk_pocket_right.append(file)
        elif 'climbing' in file.lower():
            climbing_pocket_right.append(file)
        elif 'descending' in file.lower():
            descending_pocket_right.append(file)
        elif 'desk' in file.lower():
            deskwork_pocket_right.append(file)
        elif 'lying down left' in file.lower():
            lyingLeft_pocket_right.append(file)
        elif 'lying down right' in file.lower():
            lyingRight_pocket_right.append(file)
        elif 'lying' in file.lower() and 'stomach' in file.lower():
            lyingStomach_pocket_right.append(file)
        elif 'lying' in file.lower() and 'back' in file.lower():
            lyingBack_pocket_right.append(file)
        elif 'run' in file.lower():
            run_pocket_right.append(file)
        elif 'sit' in file.lower() and 'forward' in file.lower():
            sitForward_pocket_right.append(file)
        elif 'sit' in file.lower() and 'backward' in file.lower():
            sitBackward_pocket_right.append(file)
        elif 'sit' in file.lower() and 'forward' not in file.lower() and 'backward' not in file.lower():
            sitStand_pocket_right.append(file)
    elif 'wrist_left' in file.lower():
        if 'walk' in file.lower():
            walk_wrist_left.append(file)
        elif 'climbing' in file.lower():
            climbing_wrist_left.append(file)
        elif 'descending' in file.lower():
            descending_wrist_left.append(file)
        elif 'desk' in file.lower():
            deskwork_wrist_left.append(file)
        elif 'lying down left' in file.lower():
            lyingLeft_wrist_left.append(file)
        elif 'lying down right' in file.lower():
            lyingRight_wrist_left.append(file)
        elif 'lying' in file.lower() and 'stomach' in file.lower():
            lyingStomach_wrist_left.append(file)
        elif 'lying' in file.lower() and 'back' in file.lower():
            lyingBack_wrist_left.append(file)
        elif 'run' in file.lower():
            run_wrist_left.append(file)
        elif 'sit' in file.lower() and 'forward' in file.lower():
            sitForward_wrist_left.append(file)
        elif 'sit' in file.lower() and 'backward' in file.lower():
            sitBackward_wrist_left.append(file)
        elif 'sit' in file.lower() and 'forward' not in file.lower() and 'backward' not in file.lower():
            sitStand_wrist_left.append(file)
    elif 'wrist_right' in file.lower():
        if 'walk' in file.lower():
            walk_wrist_right.append(file)
        elif 'climbing' in file.lower():
            climbing_wrist_right.append(file)
        elif 'descending' in file.lower():
            descending_wrist_right.append(file)
        elif 'desk' in file.lower():
            deskwork_wrist_right.append(file)
        elif 'lying down left' in file.lower():
            lyingLeft_wrist_right.append(file)
        elif 'lying down right' in file.lower():
            lyingRight_wrist_right.append(file)
        elif 'lying' in file.lower() and 'stomach' in file.lower():
            lyingStomach_wrist_right.append(file)
        elif 'lying' in file.lower() and 'back' in file.lower():
            lyingBack_wrist_right.append(file)
        elif 'run' in file.lower():
            run_wrist_right.append(file)
        elif 'sit' in file.lower() and 'forward' in file.lower():
            sitForward_wrist_right.append(file)
        elif 'sit' in file.lower() and 'backward' in file.lower():
            sitBackward_wrist_right.append(file)
        elif 'sit' in file.lower() and 'forward' not in file.lower() and 'backward' not in file.lower():
            sitStand_wrist_right.append(file)

class_titles = ['walk_chest_left', 'walk_chest_right', 'walk_pocket_left', 'walk_pocket_right', 'walk_wrist_left', 'walk_wrist_right',
               'climbing_chest_left', 'climbing_chest_right', 'climbing_pocket_left', 'climbing_pocket_right', 'climbing_wrist_left', 'climbing_wrist_right',
               'descending_chest_left', 'descending_chest_right', 'descending_pocket_left', 'descending_pocket_right',
               'descending_wrist_left', 'descending_wrist_right', 'deskwork_chest_left', 'deskwork_chest_right', 'deskwork_pocket_left',
               'deskwork_pocket_right', 'deskwork_wrist_left', 'deskwork_wrist_right',
               'lyingLeft_chest_left', 'lyingLeft_chest_right', 'lyingLeft_pocket_left', 'lyingLeft_pocket_right', 'lyingLeft_wrist_left', 'lyingLeft_wrist_right',
               'lyingRight_chest_left', 'lyingRight_chest_right', 'lyingRight_pocket_left', 'lyingRight_pocket_right', 'lyingRight_wrist_left', 'lyingRight_wrist_right',
               'lyingBack_chest_left', 'lyingBack_chest_right', 'lyingBack_pocket_left', 'lyingBack_pocket_right', 'lyingBack_wrist_left', 'lyingBack_wrist_right',
               'lyingStomach_chest_left', 'lyingStomach_chest_right', 'lyingStomach_pocket_left', 'lyingStomach_pocket_right', 'lyingStomach_wrist_left', 'lyingStomach_wrist_right',
               'run_chest_left', 'run_chest_right', 'run_pocket_left', 'run_pocket_right', 'run_wrist_left', 'run_wrist_right',
               'sitForward_chest_left', 'sitForward_chest_right', 'sitForward_pocket_left', 'sitForward_pocket_right', 'sitForward_wrist_left', 'sitForward_wrist_right',
               'sitBackward_chest_left', 'sitBackward_chest_right', 'sitBackward_pocket_left', 'sitBackward_pocket_right', 'sitBackward_wrist_left', 'sitBackward_wrist_right',
               'sitStand_chest_left', 'sitStand_chest_right', 'sitStand_pocket_left', 'sitStand_pocket_right', 'sitStand_wrist_left', 'sitStand_wrist_right']

classes = [walk_chest_left, walk_chest_right, walk_pocket_left, walk_pocket_right, walk_wrist_left, walk_wrist_right,
           climbing_chest_left, climbing_chest_right, climbing_pocket_left, climbing_pocket_right, climbing_wrist_left, climbing_wrist_right,
           descending_chest_left, descending_chest_right, descending_pocket_left, descending_pocket_right,
           descending_wrist_left, descending_wrist_right, deskwork_chest_left, deskwork_chest_right, deskwork_pocket_left,
           deskwork_pocket_right, deskwork_wrist_left, deskwork_wrist_right,
           lyingLeft_chest_left, lyingLeft_chest_right, lyingLeft_pocket_left, lyingLeft_pocket_right, lyingLeft_wrist_left, lyingLeft_wrist_right,
           lyingRight_chest_left, lyingRight_chest_right, lyingRight_pocket_left, lyingRight_pocket_right, lyingRight_wrist_left, lyingRight_wrist_right,
           lyingBack_chest_left, lyingBack_chest_right, lyingBack_pocket_left, lyingBack_pocket_right, lyingBack_wrist_left, lyingBack_wrist_right,
           lyingStomach_chest_left, lyingStomach_chest_right, lyingStomach_pocket_left, lyingStomach_pocket_right, lyingStomach_wrist_left, lyingStomach_wrist_right,
           run_chest_left, run_chest_right, run_pocket_left, run_pocket_right, run_wrist_left, run_wrist_right,
           sitForward_chest_left, sitForward_chest_right, sitForward_pocket_left, sitForward_pocket_right, sitForward_wrist_left, sitForward_wrist_right,
           sitBackward_chest_left, sitBackward_chest_right, sitBackward_pocket_left, sitBackward_pocket_right, sitBackward_wrist_left, sitBackward_wrist_right,
           sitStand_chest_left, sitStand_chest_right, sitStand_pocket_left, sitStand_pocket_right, sitStand_wrist_left, sitStand_wrist_right]

def make_1d(x, y, z):
    result_vector = []
    for i in range(len(x)):
        result_vector.append(x[i])
        result_vector.append(y[i])
        result_vector.append(z[i])
    return result_vector

the_big_x = []
the_big_y = []

for i in range(len(classes)):
    print(class_titles[i]+'==================================================================================>>>')
    for file in classes[i]:
        if '2019' in file:
            print(file)
            temp = pd.read_csv(file, skiprows=11)
            # print(temp)
            simple_vector = make_1d(temp['accel_x'], temp['accel_y'], temp['accel_z'])
            print(simple_vector)
            print(len(simple_vector))
            the_big_x.append(simple_vector)
            the_big_y.append(class_titles[i])
        elif '2020' in file:
            print(file)
            temp = pd.read_csv(file, skiprows=5)
            # print(temp)
            simple_vector = make_1d(temp['accel_x'], temp['accel_y'], temp['accel_z'])
            print(simple_vector)
            print(len(simple_vector))
            the_big_x.append(simple_vector)
            the_big_y.append(class_titles[i])

print(len(the_big_x))
print(len(the_big_y))
the_data = pd.DataFrame({'value_vector':the_big_x, 'class':the_big_y})
the_data.to_csv('training_data.csv')








