from walker import BiomechanicsTools

# --- Options --- #
data_path = "C:\\Users\\felie\\Downloads\\LAO"
kinematic_model_file_path = "C:\\Users\\felie\\Downloads\\LAO\\wholebody.bioMod" #"walker/LAO.bioMod"
static_trial = f"{data_path}\\LAO_01_Statique_2.c3d"
trials = (
    f"{data_path}\\LAO_01_Cond0002_02.c3d",
)

print(kinematic_model_file_path)
print('****')
# --------------- #


def main():
    print(kinematic_model_file_path)
    # Generate the personalized kinematic model
    tools = BiomechanicsTools(body_mass=58, include_upper_body=True)
    tools.personalize_model(static_trial, kinematic_model_file_path)

    # Perform some biomechanical computation
    for trial in trials:
        print(trial)
        tools.process_trial(trial, compute_automatic_events=False)

    # TODO: Bioviz vizual bug with the end of the trial when resizing the window
    # TODO: Record a tutorial


if __name__ == "__main__":
    main()
