from walker import BiomechanicsTools

# --- Options --- #
data_path = "D:\\Data\\Sujet 10\\Vicon"
kinematic_model_file_path = "Sujet10.bioMod"
static_trial = f"{data_path}\\Sujet10 Cal 01.c3d"
trials = (
    f"{data_path}\\Sujet10_CAS_GAS1.c3d",
    f"{data_path}\\Sujet10_CSS_GAS1.c3d",
)

print(kinematic_model_file_path)
print('****')
# --------------- #


def main():
    print(kinematic_model_file_path)
    # Generate the personalized kinematic model
    tools = BiomechanicsTools(body_mass=100, include_upper_body=True)
    tools.personalize_model(static_trial, kinematic_model_file_path)

    # Perform some biomechanical computation
    for trial in trials:
        print(trial)
        tools.process_trial(trial, compute_automatic_events=False)

    # TODO: Bioviz vizual bug with the end of the trial when resizing the window
    # TODO: Record a tutorial

if __name__ == "__main__":
    main()
