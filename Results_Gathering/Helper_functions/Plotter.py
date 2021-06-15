import os


def __get_directory__(*args):
    plotter_file = os.path.abspath(__file__)
    base_path = plotter_file.split("Results_Gathering")[0]
    for arg in args:
        base_path = os.path.join(base_path, arg)
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    return base_path


def write_gathered_scores(scores, title="Scores Baseline"):
    folders = "Results", "Standard Model"
    folder = __get_directory__(*folders)
    print(folder)
    for key in scores.keys():
        file_title = "{} - Condition {}.csv".format(title, key)
        file_location = os.path.join(folder, file_title)
        r2_sc, mse_sc, phm_sc = scores[key]['r2'], scores[key]['mse'], scores[key]['phm']
        with open(file_location, 'w+') as file:
            file.write("epoch;r2;mse;phm\n")
            for epoch in range(len(r2_sc)):
                file.write("{};{};{};{}\n".format(epoch, r2_sc[epoch], mse_sc[epoch], phm_sc[epoch]))
