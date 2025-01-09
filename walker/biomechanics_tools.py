import itertools
import os.path
from scipy.signal import butter, filtfilt, savgol_filter
import biorbd
from biorbd.model_creation import C3dData
# import bioviz
import ezc3d
import numpy as np
from scipy import signal
from .misc import differentiate, to_rotation_matrix, to_euler
from .plugin_gait import SimplePluginGait
import scipy.io


def suffix_to_all(values: tuple[str, ...] | list[str, ...], suffix: str) -> tuple[str, ...]:
    return tuple(f"{n}{suffix}" for n in values)


class BiomechanicsTools:
    def __init__(self, model_type):
        self.generic_model = model_type
        self.model = None #biorbd.Model("C:\\Users\\felie\\PycharmProjects\\walkerKinematicReconstruction\\walker\\49mks.bioMod")

        self.is_kinematic_reconstructed: bool = False
        self.is_inverse_dynamic_performed: bool = False
        self.c3d_path: str | None = None
        self.c3d: ezc3d.c3d | None = None
        self.t: np.ndarray = np.ndarray(())
        self.q: np.ndarray = np.ndarray(())
        self.qdot: np.ndarray = np.ndarray(())
        self.qddot: np.ndarray = np.ndarray(())
        self.tau: np.ndarray = np.ndarray(())
        self.center_of_mass = np.ndarray(())
        self.force = np.ndarray(())
        self.cop = np.ndarray(())
        self.moment = np.ndarray(())

        self.events = None
        self.bioviz_window: bioviz.Viz | None = None

    @property
    def is_model_loaded(self):
        return self.model is not None

    def _compute_center_of_mass(self) -> np.ndarray:
        if not self.is_model_loaded:
            raise RuntimeError("The biorbd model must be loaded. You can do so by calling generate_personalized_model")

        self.com = np.array([self.model.CoM(q).to_array() for q in self.q.T]).T
        return self.com

    def forcedatafilter(self, data, order, sampling_rate, cutoff_freq):
        # Normaliser la fréquence de coupure par rapport à la fréquence de Nyquist
        nyquist_rate = sampling_rate / 2.0
        normalized_cutoff = cutoff_freq / nyquist_rate

        # Concevoir le filtre Butterworth
        b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
        filtered_x=np.zeros([len(data[:, 0]), len(data[0, :])])
        # Appliquer le filtre aux données
        for axe in range(len(filtered_x)):
            filtered_x[axe, :] = signal.filtfilt(b, a, data[axe, :])
        """
        nyquist_rate = sampling_rate / 2.0
        normalized_cutoff = 5 / nyquist_rate
        b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)
        filtered_x2 = np.zeros([len(data[:, 0]), len(data[0, :])])
        # Appliquer le filtre aux données
        for axe in range(len(filtered_x)):
            filtered_x2[axe, :] = signal.filtfilt(b, a, filtered_x[axe, :])
        """
        return filtered_x

    def normalize(self, v):
        """ Normalize a vector. """
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def personalize_model(self, static_trial: str, model_path: str = "temporary.bioMod"):
        """
        Collapse the generic model according to the data of the static trial

        Parameters
        ----------
        static_trial
            The path of the c3d file of the static trial to create the model from
        model_path
            The path of the generated bioMod file
        """

        self.generic_model.write(save_path=model_path, data=C3dData(static_trial))
        self.model = biorbd.Model(model_path)

    def process_trial(self, trial: str, compute_automatic_events: bool = False) -> None:
        """
        Performs everything to do with a specific trial, including kinematic reconstruction and export

        Parameters
        ----------
        trial
            The path to the c3d file of the trial to reconstruct the kinematics from
        compute_automatic_events
            If the automatic event finding algorithm should be used. Otherwise, the events in the c3d file are used
        """
        self.process_kinematics(trial,visualize = False)
        self.inverse_dynamics(trial)

        # Write the c3d as if it was the plug in gate output
        path = os.path.dirname(trial)
        file_name = os.path.splitext(os.path.basename(trial))[0]
        self.to_c3d(f"{path}/{file_name}_processed2.c3d", compute_automatic_events=compute_automatic_events)

    def process_kinematics(self, trial: str, visualize: bool = False):
        """
        Performs the kinematics reconstruction

        Parameters
        ----------
        trial
            The path to the c3d file of the trial to reconstruct the kinematics from
        visualize
            If the reconstruction should be shown

        Returns
        -------
        This method populates self.c3d, self.c3d_path, self.t, self.q, self.qdot and self.qddot
        """
        self.load_c3d_file(trial)
        frames = self._select_frames_to_reconstruct(acceptance_threshold=0.7)
        self.reconstruct_kinematics(frames=frames)
        self.unwrap_kinematics()
        if visualize==True:
            self.show_kinematic_reconstruction()

    def load_c3d_file(self, trial):
        """
        Load the c3d in the variables

        Parameters
        ----------
        trial
            The path to the c3d file of the trial to reconstruct the kinematics from
        """
        self.c3d_path = trial
        self.c3d = ezc3d.c3d(self.c3d_path, extract_forceplat_data=True)

    def _select_frames_to_reconstruct(self, acceptance_threshold: float = 0.7) -> slice:
        """
        Select the first and last frame where the percentage threshold of visible markers is satisfied

        Parameters
        ----------
        acceptance_threshold
            The percentage of markers that must be present. Default all the marker should be seen

        Returns
        -------
        The frames to reconstruct
        """
        technical_markers = tuple(name.to_string() for name in self.model.technicalMarkerNames())
        n_technical_markers = len(technical_markers)
        c3d_marker_names = self.c3d["parameters"]["POINT"]["LABELS"]["value"]
        marker_index = tuple(c3d_marker_names.index(n) for n in technical_markers)
        n_frames = self.c3d["data"]["points"].shape[2]

        n_nan_marker_per_frame = np.sum(
            np.isnan(np.sum(self.c3d["data"]["points"][:, marker_index, :], axis=0)), axis=0
        )
        missing_percentage_per_frame = n_nan_marker_per_frame / n_technical_markers

        is_frame_accepted = list(missing_percentage_per_frame < (1 - acceptance_threshold))
        first_frame = is_frame_accepted.index(True)
        is_frame_accepted.reverse()
        last_frame = n_frames - is_frame_accepted.index(True)

        return slice(first_frame, last_frame)

    def reconstruct_kinematics(self, frames: slice = slice(None)) -> np.ndarray:
        """
        Reconstruct the kinematics of the specified trial assuming a biorbd model is loaded using a Kalman filter

        Parameters
        ----------
        frames
            The frames to reconstruct

        Returns
        -------
        The matrix nq x ntimes of the reconstructed kinematics
        """
        if not self.is_model_loaded:
            raise RuntimeError("The biorbd model must be loaded. You can do so by calling generate_personalized_model")

        first_frame_c3d = self.c3d["header"]["points"]["first_frame"]
        last_frame_c3d = self.c3d["header"]["points"]["last_frame"]
        n_frames_before = 0#(frames.start - first_frame_c3d) if frames.start is not None else 0
        n_frames_after = 0#(last_frame_c3d - frames.stop + 1) if frames.stop is not None else 0
        n_frames_total = last_frame_c3d - first_frame_c3d + 1
        """
        dofred = np.arange(27,44, 1)
        valuesdof = [0.26, 1.4, 0.35, 0.15, 2.5, 0.35, 0.52, 0.42, 0.52, 0.52,0.26, 1.4, 0.35, 0.15, 2.5, 0.35, 0.52, 0.42, 0.52, 0.52]
        qmax = np.zeros([44])
        for i, dof in enumerate(dofred):
            qmax[dof] = valuesdof[i]  # Mettre à jour la limite maximale pour chaque DoF
        
        self.generic_model.setQRangeMin(-qmax)
        self.generic_model.setQRangeMax(qmax)
        # Sauvegarde du modèle mis à jour
        self.generic_model.Write("mon_modele_avec_rangesQ.bioMod")
        """
        self.t, self.q, self.qdot, self.qddot = biorbd.extended_kalman_filter(self.model, self.c3d_path, frames=frames)
        # Align the data with the c3d
        n_q = self.q.shape[0]
        dof_padding_before = np.zeros((n_q, n_frames_before))
        dof_padding_after = np.zeros((n_q, n_frames_after))
        self.t = np.linspace(first_frame_c3d, last_frame_c3d, n_frames_total)
        self.q = np.concatenate((dof_padding_before, self.q, dof_padding_after), axis=1)
        self.qdot = np.concatenate((dof_padding_before, self.qdot, dof_padding_after), axis=1)
        self.qddot = np.concatenate((dof_padding_before, self.qddot, dof_padding_after), axis=1)

        self.is_kinematic_reconstructed = True

        return self.q

    def relative_to_vertical(self, segment: str, angle_sequence: str, q: np.ndarray = None) -> np.array:
        """
        Provide the Euler angles of the specified segment relative to vertical

        Parameters
        ----------
        segment
            The name of the segment to express relative to vertical
        angle_sequence
            The sequence of angle to reconstruct to
        q
            The generalized coordinates to use. If None is sent, then self.q is
            used (assuming kinematics was reconstructed)

        Returns
        -------
        The Euler angles for the specified segment
        """
        if q is None and not self.is_kinematic_reconstructed:
            raise RuntimeError("The kinematics must be reconstructed before performing the unwrap")
        q = self.q if q is None else q

        segment_idx = tuple(s.name().to_string() for s in self.model.segments()).index(segment)
        jcs = np.ndarray((4, 4, self.q.shape[1]))
        for i, q in enumerate(q.T):
            jcs[:, :, i] = self.model.globalJCS(q, segment_idx).to_array()
        return to_euler(jcs, angle_sequence)

    def unwrap_kinematics(self):
        """
        Performs unwrap on the kinematics from which it re-expressed in terms of matrix rotation before
        (which makes it more likely to be in the same quadrant)

        Returns
        -------

        """

        if not self.is_kinematic_reconstructed:
            raise RuntimeError("The kinematics must be reconstructed before performing the unwrap")

        segment_names = tuple(s.name().to_string() for s in self.model.segments())
        dof_names = tuple(n.to_string() for n in self.model.nameDof())
        for segment_name in segment_names:
            segment = self.model.segment(segment_names.index(segment_name))
            angle_sequence = segment.seqR().to_string()
            if not angle_sequence:
                continue
            angle_names = tuple(
                segment.nameDof(i).to_string()
                for i in range(segment.nbDofTrans(), segment.nbDofTrans() + segment.nbDofRot())
            )
            angle_index = tuple(dof_names.index(f"{segment_name}_{angle_name}") for angle_name in angle_names)

            data = self.q[angle_index, :]
            rot = to_rotation_matrix(angles=data, angle_sequence=angle_sequence)
            self.q[angle_index, :] = np.unwrap(to_euler(rot, angle_sequence), axis=1)

    def get_cycles(self, side) -> tuple[int, ...]:
        """
        Get the cycles slices based on the C3D file. More specifically,
        it returns the all the indices of the Foot Strikes

        Parameters
        ----------
        side
            The side ("right" or "left") to get the slices from

        Returns
        -------
        All the cycles
        """
        if not self.c3d_path:
            raise RuntimeError("A C3D file must be loaded")

        if side != "Right" and side != "Left":
            raise ValueError("side must be 'Right' or 'Left'")

        events_side = self.c3d["parameters"]["EVENT"]["CONTEXTS"]["value"]
        events_tag = self.c3d["parameters"]["EVENT"]["LABELS"]["value"]
        events_time = self.c3d["parameters"]["EVENT"]["TIMES"]["value"][1, :]

        rate = self.c3d["header"]["points"]["frame_rate"]
        first_time = self.c3d["header"]["points"]["first_frame"] / rate
        last_time = self.c3d["header"]["points"]["last_frame"] / rate
        t = np.linspace(first_time, last_time, self.c3d["data"]["points"].shape[2])
        events_index = [list(t > event).index(True) for event in events_time]

        out = []
        for event_side, event_tag, event_index in zip(events_side, events_tag, events_index):
            if event_side != side:
                continue
            if event_tag != "Foot Strike":
                continue
            out.append(event_index)

        return tuple(out)

    def show_kinematic_reconstruction(self):
        """
        Opens a BioViz window and show the kinematic reconstruction
        """

        if not self.is_kinematic_reconstructed:
            raise RuntimeError("The kinematics must be reconstructed before showing the reconstruction")

        viz = bioviz.Viz(loaded_model=self.model)
        viz.load_movement(self.q)
        viz.load_experimental_markers(self.c3d_path)
        viz.radio_c3d_editor_model.click()
        viz.exec()

    def inverse_dynamics(self, trial) -> np.ndarray:
        """
        Performs the inverse dynamics of a previously reconstructed kinematics

        Returns
        -------
        Stores and return de generalized forces
        """
        if not self.is_kinematic_reconstructed:
            raise RuntimeError("The kinematics must be reconstructed before performing the inverse dynamics")

        ContactName = ["LFoot", "RFoot"]
        tau_data = []
        f_extfilt = np.zeros([len(ContactName), 3, 20 * len(self.q[0, :])])
        moment_extfilt = np.zeros([len(ContactName), 3, 20 * len(self.q[0, :])])
        moment_origin = np.zeros([len(ContactName), 3, 20 * len(self.q[0, :])])
        cop_extfilt = np.zeros([len(ContactName), 3, 20 * len(self.q[0, :])])
        originPf = np.zeros([len(ContactName), 3, 1])
        for contact in range(len(ContactName)):
            f_ext = self.c3d['data']['platform'][contact]['force']
            f_extfilt[contact, :, :] = self.forcedatafilter(f_ext, 4, 2000, 20)
            moment_ext = self.c3d['data']['platform'][contact]['moment'] / 1000
            moment_extfilt[contact, :, :] = self.forcedatafilter(moment_ext, 4, 2000, 20)
            cop_ext = self.c3d['data']['platform'][contact]['center_of_pressure'] / 1000
            cop_extfilt[contact, :, :] = self.forcedatafilter(cop_ext, 4, 2000, 10)
            originPf[contact, 0, 0] = np.mean(self.c3d['data']['platform'][contact]['corners'][0])/1000
            originPf[contact, 1, 0] = np.mean(self.c3d['data']['platform'][contact]['corners'][1])/1000
            originPf[contact, 2, 0] = np.mean(self.c3d['data']['platform'][contact]['corners'][2])/1000

            for ii in range(len(moment_extfilt[contact, 0, :])):
                r = originPf[contact, :, 0]-cop_extfilt[contact, :, ii]
                M_offset = np.cross(r, f_extfilt[contact, :, ii])
                moment_origin[contact, :, ii] = moment_extfilt[contact, :, ii] + M_offset

        moment_extfilt = moment_origin
        #moment_extfilt = np.zeros([len(ContactName), 3, 20 * len(self.q[0, :])])

        self.force = np.empty([2, 9, len(self.q[0, :])])
        PointApplication = np.zeros([2, 3, len(self.q[0, :])])

        # Smoothing position data
        self.q = savgol_filter(self.q, 30, 3)

        # Initialize arrays for angular velocity and acceleration
        angular_velocity = np.empty_like(self.q)
        angular_acc = np.empty_like(self.q)

        # Calculate angular velocity by taking the gradient of the smoothed position data
        for dof in range(len(self.q[:, 0])):
            angular_velocity[dof, :] = np.gradient(self.q[dof, :], 1 / 100)

        # Apply filtering to the angular velocity data
        self.qdot = self.forcedatafilter(angular_velocity, 4, 100, 10)

        # Calculate angular acceleration by taking the gradient of the filtered velocity data
        for dof in range(len(self.q[:, 0])):
            angular_acc[dof, :] = np.gradient(self.qdot[dof, :], 1 / 100)

        # Apply filtering to the angular acceleration data
        self.qddot = self.forcedatafilter(angular_acc, 4, 100, 10)


        for i in range(len(self.q[0, :])):
            self.ext_load = self.model.externalForceSet()

            for contact in range(len(ContactName)):
                name = biorbd.String(ContactName[contact])
                spatial_vector = np.concatenate((moment_extfilt[contact, :, 20 * i], f_extfilt[contact, :, 20 * i]))
                PointApplication[contact, :, i] = cop_extfilt[contact, :, 20 * i] #self.c3d['data']['platform'][contact]['origin']
                PA = PointApplication[contact, :, i]
                if spatial_vector[5] > 5:
                    if (PointApplication[contact, 2, i-1] or abs(PA[2]-PointApplication[contact, 2, i-1]) < 0.00001):
                        self.ext_load.add(name, spatial_vector, PA)
                        self.force[contact, 0:3, i] = PA
                        self.force[contact, 3:6, i] = f_extfilt[contact, :, 20 * i]
                        self.force[contact, 6:, i] = moment_extfilt[contact, :, 20 * i]

            tau = self.model.InverseDynamics(self.q[:, i], self.qdot[:, i], self.qddot[:, i], self.ext_load)
            tau_data.append(tau.to_array())

        tau_data = np.array(tau_data)
        self.tau = np.transpose(tau_data)
        #self.tau = self.forcedatafilter(angular_acc, 1, 100, 15)
        self.is_inverse_dynamic_performed = True
        return self.tau

    def find_feet_events(self) -> tuple[int, tuple[str, ...], tuple[str, ...], np.ndarray]:
        """
        Returns
        -------
        number of events: int
            The number of events
        event_contexts: tuple[str, ...]
            If a specific event arrived the on 'Left' or on the 'Right'
        event_labels: tuple[str, ...]
            If a specific event is a 'Foot Strike' or a Foot Off'
        event_times: np.ndarray
            The time for a specific event. the first row should be all zeros for some unknown reason
        """

        def find_foot_events(heel_marker_name: str, toe_marker_name: str):
            """
            Finds the events where the foot interacts with the ground. The return is expected to have an equal number
            of heel strikes and toe off.

            The algorithm is to take the lowest velocity of the heel, then to find the first time this velocity hits 0;
            this is the heel strike.
            The maximum heel velocity after that point is prior to the toe off and the highest toe velocity is just
            after. The toe off is therefore 80% of that distance towards the max velocity

            Parameters
            ----------
            heel_marker_name
                The name of the heel marker in the model
            toe_marker_name
                The name of the toe marker in the model

            Returns
            -------
            The 'heel strikes' and 'toe off' event. The number of each is expected to be equal
            """

            markers = biorbd.markers_to_array(self.model, self.q)
            heel_idx = biorbd.marker_index(self.model, heel_marker_name)
            toe_idx = biorbd.marker_index(self.model, toe_marker_name)
            heel_height = markers[(2,), heel_idx, :]
            heel_velocity = differentiate(heel_height, self.t[1] - self.t[0])
            toe_height = markers[(2,), toe_idx, :]
            toe_velocity = differentiate(toe_height, self.t[1] - self.t[0])
            toe_acceleration = differentiate(toe_velocity, self.t[1] - self.t[0])

            idx_peaks_heel_strike = []
            idx_peak_pre_heel_strike = signal.find_peaks(-heel_velocity[0, :], height=0.5)[0]
            for heel_idx in idx_peak_pre_heel_strike:
                # find the first time the signal crosses 0 from that lowest point
                idx_peaks_heel_strike.append(heel_idx + np.argmax(np.diff(np.sign(heel_velocity[:, heel_idx:])) != 0))

            idx_peaks_toe_off = []
            t_peaks_pre_toe_off = signal.find_peaks(heel_velocity[0, :], height=0.5)[0]
            for toe_idx in t_peaks_pre_toe_off:
                # find the first time the signal crosses 0 from that lowest point
                idx_peaks_post_toe_off = np.argmax(np.diff(np.sign(toe_acceleration[:, toe_idx:])) != 0)
                idx_peaks_toe_off.append(int(toe_idx + 0.8 * idx_peaks_post_toe_off))

            # Associate each heel strike with its toe off
            first_toe_off_idx = -1
            for i, toe in enumerate(idx_peaks_toe_off):
                if toe > idx_peaks_heel_strike[0]:
                    first_toe_off_idx = i
                    break
            last_heel_strike_idx = -1
            for i, heel in enumerate(reversed(idx_peaks_heel_strike)):
                if heel < idx_peaks_toe_off[-1]:
                    last_heel_strike_idx = len(idx_peaks_heel_strike) - i
                    break

            if first_toe_off_idx == -1 or last_heel_strike_idx == -1:
                Warning("No heel strikes that correspond to the toe offs were found")

            heel_strikes = idx_peaks_heel_strike[:last_heel_strike_idx]
            toe_off = idx_peaks_toe_off[first_toe_off_idx:]
            if len(heel_strikes) != len(toe_off):
                Warning("The number of heel strikes and toe off does not match")

            return heel_strikes, toe_off

        left_foot_events = find_foot_events("LHEE", "LTOE")
        right_foot_events = find_foot_events("RHEE", "RTOE")
        # From that point, it is assumed that `len(events[0]) == len(events[1])`, that is there are equal number of
        # foot strikes and toe off

        events_number = (len(left_foot_events[0]) + len(right_foot_events[0])) * 2
        events_contexts = ("Left",) * len(left_foot_events[0]) * 2 + ("Right",) * len(right_foot_events[0]) * 2
        events_labels = ("Foot Strike", "Foot Off") * int(events_number / 2)
        events_times = np.array(
            (
                (0.0,) * events_number,
                self.t[
                    np.array(
                        tuple(
                            itertools.chain(  # flatten the left/right, heel strike/toe off
                                *[[heel, toe] for heel, toe in zip(*left_foot_events)]
                                + [[heel, toe] for heel, toe in zip(*right_foot_events)]
                            )
                        )
                    )
                ],
            )
        )
        return events_number, events_contexts, events_labels, events_times

    def _dispatch_events_from_bioviz(self):
        self.events = [self.bioviz_window.n_events]
        self.events += self.bioviz_window.analyses_c3d_editor.convert_event_for_c3d(
            self.c3d["header"]["points"]["frame_rate"]
        )

        self.bioviz_window.vtk_window.close()

    def to_c3d(self, save_path: str, compute_automatic_events: bool = False) -> None:
        """
        Create a Nexus-like c3d file from the reconstructed kinematics

        Parameters
        ----------
        save_path
            The path where to save the c3d
        compute_automatic_events
            If the automatic event finding algorithm should be used. Otherwise, the events in the c3d file are used
        """

        if not self.is_kinematic_reconstructed:
            raise RuntimeError(
                "Kinematics should be reconstructed before writing to c3d. " "Please call 'kinematic_reconstruction'"
            )

        c3d = ezc3d.c3d()

        # Fill it with points, angles, power, force, moment
        c3d["parameters"]["POINT"]["RATE"]["value"] = [int(self.c3d["parameters"]["POINT"]["RATE"]["value"][0])]
        c3d.add_parameter("POINT", "ANGLE_UNITS", ["deg"])
        point_names = [name.to_string() for name in self.model.markerNames()]
        point_names.extend(["CentreOfMass", "CentreOfMassFloor", "CoP1", "CoP2","force1","force2","moment1","moment2"])
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Angles"))
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Vitesse"))
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Acc"))
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Power"))
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Force"))
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Moment"))
        c3d["parameters"]["POINT"]["UNITS"] = self.c3d["parameters"]["POINT"]["UNITS"]

        # Transfer the marker data to the new c3d
        c3d["parameters"]["POINT"]["LABELS"]["value"] = point_names
        first_frame = self.c3d["header"]["points"]["first_frame"]
        last_frame = self.c3d["header"]["points"]["last_frame"]
        n_frame = last_frame - first_frame + 1
        data = np.ndarray((4, len(point_names), n_frame)) * np.nan
        data[3, ...] = 1
        for i, name_in_c3d in enumerate(self.c3d["parameters"]["POINT"]["LABELS"]["value"]):
            if name_in_c3d[0] == "*" or name_in_c3d not in point_names:
                continue
            # Make sure it is in the right order
            data[:, point_names.index(name_in_c3d), :] = self.c3d["data"]["points"][:, i, :]

        self._compute_center_of_mass()
        data[:3, point_names.index("CentreOfMass"), :] = self.com
        data[:3, point_names.index("CentreOfMassFloor"), :] = self.com
        data[2, point_names.index("CentreOfMassFloor"), :] = 0

        data[:3, point_names.index("CoP1"), :] = self.force[0,:3,:]
        data[:3, point_names.index("CoP2"), :] = self.force[1,:3,:]
        data[:3, point_names.index("force1"), :] = self.force[0, 3:6, :]
        data[:3, point_names.index("force2"), :] = self.force[1, 3:6, :]
        data[:3, point_names.index("moment1"), :] = self.force[0, 6:, :]
        data[:3, point_names.index("moment2"), :] = self.force[1, 6:, :]
        """
        a=data[:3, point_names.index("CoP1"), :]
        b=data[:3, point_names.index("CoP2"), :]
        matrix1 =np.expand_dims(a, axis=0)
        matrix2 = np.expand_dims(b, axis=0)
        grouped_matrix1 = np.vstack((matrix1, matrix2))
        a = data[:3, point_names.index("force1"), :]
        b = data[:3, point_names.index("force2"), :]
        matrix1 = np.expand_dims(a, axis=0)
        matrix2 = np.expand_dims(b, axis=0)
        grouped_matrix2 = np.vstack((matrix1, matrix2))
        grouped_matrix = np.concatenate((grouped_matrix1 , grouped_matrix2), axis=1)
        """
        # Dispatch the kinematics and kinematics
        for dof, idx in self.generic_model.dof_index.items():
            if idx is None:
                continue
            data[:3, point_names.index(f"{dof}Angles"), :] = self.q[idx, :] * 180 / np.pi
            data[:3, point_names.index(f"{dof}Vitesse"), :] = self.qdot[idx, :]
            data[:3, point_names.index(f"{dof}Acc"), :] = self.qddot[idx, :]
            data[:3, point_names.index(f"{dof}Moment"), :] = self.tau[idx, :]
            data[:3, point_names.index(f"{dof}Power"), :] = self.tau[idx, :] * self.qdot[idx, :]
        c3d["data"]["points"] = data

        self.bioviz_window = bioviz.Viz(loaded_model=self.model)
        self.bioviz_window.load_movement(self.q)
        self.bioviz_window.load_experimental_forces(self.force[:,:6,:], segments=['Ground', 'Ground'], normalization_ratio=0.5)
        self.bioviz_window.load_experimental_markers(self.c3d_path)
        self.bioviz_window.radio_c3d_editor_model.click()

        if compute_automatic_events:
            self.bioviz_window.clear_events()
            # Find and add events
            self.events = self.find_feet_events()
            events_number, events_contexts, events_labels, events_times = self.events
            for context, label, time in zip(events_contexts, events_labels, events_times[1, :]):
                frame = (
                    int(time * self.c3d["header"]["points"]["frame_rate"]) - self.c3d["header"]["points"]["first_frame"]
                )
                self.bioviz_window.set_event(frame, f"{context} {label}")
        self.bioviz_window.analyses_c3d_editor.export_c3d_button.disconnect()
        self.bioviz_window.analyses_c3d_editor.export_c3d_button.clicked.connect(self._dispatch_events_from_bioviz)
        self.bioviz_window.exec()
        if self.events is None:
            raise RuntimeError("No events found, have you clicked Export C3D?")
        events_number, events_contexts, events_labels, events_times = self.events

        c3d.add_parameter("EVENT", "USED", (events_number,))
        c3d.add_parameter("EVENT", "CONTEXTS", events_contexts)
        c3d.add_parameter("EVENT", "LABELS", events_labels)
        c3d.add_parameter("EVENT", "TIMES", events_times)

        # Copy the header
        for element in self.c3d["header"]:
            for item in self.c3d["header"][element]:
                c3d["header"][element][item] = self.c3d["header"][element][item]

        # Dispatch the analog and force_plate data
        c3d["parameters"]["ANALOG"] = self.c3d["parameters"]["ANALOG"]
        for i, label in enumerate(self.c3d["parameters"]["ANALOG"]["LABELS"]["value"]):
            label = label.replace("Force.", "")
            label = label.replace("Moment.", "")
            c3d["parameters"]["ANALOG"]["LABELS"]["value"][i] = label

        c3d["parameters"]["FORCE_PLATFORM"] = self.c3d["parameters"]["FORCE_PLATFORM"]
        c3d["data"]["analogs"] = self.c3d["data"]["analogs"]

        # Write the data
        c3d.write(save_path)

