from pyorerun import BiorbdModel, PhaseRerun
import numpy as np

participant_name = "VIF_04"
trial_names = ["Cond0001"]
data_path = "data"
body_mass = 71  # Kg

kinematic_model_file_path = f"{data_path}/{participant_name}.bioMod"
static_trial = f"{data_path}/{participant_name}_Statique.c3d"
trials = [
    f"{data_path}/{participant_name}_{condition_name}.c3d" for condition_name in trial_names
]

nb_frames = 10
nb_seconds = 0.1
t_span = np.linspace(0, nb_seconds, nb_frames)

model = BiorbdModel(f"/home/charbie/Documents/Programmation/walkerKinematicReconstruction/{data_path}/{participant_name}.bioMod")
q = np.zeros((model.model.nbQ(), nb_frames))

viz = PhaseRerun(t_span)
viz.add_animated_model(model, q)
viz.rerun("msk_model")

