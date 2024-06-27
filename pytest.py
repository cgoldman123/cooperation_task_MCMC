
subject_list_path = 'L:/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/cooperation_prolific_IDs.csv'

subjects = []
with open(subject_list_path) as infile:
    for line in infile:
        if 'id' not in line:
            subjects.append(line.strip())

subjects_string = ",".join(subjects)
print(subjects_string)