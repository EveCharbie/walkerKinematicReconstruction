from biorbd.model_creation import (
    Segment,
    Contact,
    Translations,
    Rotations,
    Ranges,
    MuscleGroup,
    Muscle,
    MuscleType,
    MuscleStateType,
)
import numpy as np
import biorbd

from walker.plugin_gait import SimplePluginGait


standard_model = {"model": biorbd.Model("data/Gait_1leg_12dof_flatfoot.bioMod")}
standard_model["marker_names"] = [m.to_string() for m in standard_model["model"].markerNames()]
standard_model["markers"] = {m_name: standard_model["model"].marker(standard_model["marker_names"].index(m_name)).to_array() for m_name in standard_model["marker_names"]}


class OCPPluginGait(SimplePluginGait):
    """
    This is a modified version of the Plug-in Gait that can be used in optimal control problems.
    Main differences are:
    - DoF removed:
    - Muscles added: plantar flexors () and extensors ()
    - Added ranges of motion for each DoF based on (Maldonado et al., 2018: Whole-body musculo-skeletal model V1)
    """

    def __init__(
        self,
        body_mass: float,
        shoulder_offset: float = None,
        elbow_width: float = None,
        wrist_width: float = None,
        hand_thickness: float = None,
        leg_length: dict[str, float] = None,
        ankle_width: float = None,
        include_upper_body: bool = True,
    ):
        """
        Parameters
        ----------
        body_mass
            The mass of the full body
        shoulder_offset
            The measured shoulder offset of the subject. If None is provided, it is approximated using
            Rab (2002), A method for determination of upper extremity kinematics
        elbow_width
            The measured width of the elbow. If None is provided 115% of the distance between WRA and WRB is used
        wrist_width
            The measured width of the wrist. If None is provided, 2cm is used
        hand_thickness
            The measured thickness of the hand. If None is provided, 1cm is used
        leg_length
            The measured leg length in a dict["R"] or dict["L"]. If None is provided, the 95% of the ASI height is
            used (therefore assuming the subject is standing upright during the static trial)
        ankle_width
            The measured ankle width. If None is provided, the distance between ANK and HEE is used.
        include_upper_body
            If the upper body should be included in the reconstruction (set all the technical flag of the upper body
            marker false if not included)

        Since more markers are used in our version (namely Knee medial and ankle medial), the KJC and AJC were
        simplified to be the mean of these markers with their respective lateral markers. Hence, 'ankle_width'
        is no more useful
        """
        super(OCPPluginGait, self).__init__(body_mass,
                                            shoulder_offset,
                                            elbow_width,
                                            wrist_width,
                                            hand_thickness,
                                            leg_length,
                                            ankle_width,
                                            include_upper_body)
        self._modify_kinematic_model()
        self._define_muscle_model()


    def _get_foot_characteristics(self, m):

        # width = R_FM5 -> R_FM1
        R_FM5 = standard_model["markers"]["R_FM5"]
        R_FM1 = standard_model["markers"]["R_FM1"]
        foot_width_standard = np.linalg.norm(R_FM5 - R_FM1)
        foot_width_real = (np.linalg.norm(np.nanmean(m["RMFH5"], axis=1) - np.nanmean(m["RMFH1"], axis=1)) + np.linalg.norm(np.nanmean(m["LMFH5"], axis=1) - np.nanmean(m["LMFH1"], axis=1))) / 2
        foot_width_ratio = foot_width_real / foot_width_standard

        # length = mid(R_FM5, R_FM1) - R_FCC
        R_FCC = standard_model["markers"]["R_FCC"]
        foot_length_standard = np.linalg.norm((R_FM5 + R_FM1)/2 - R_FCC)
        foot_length_real = (np.linalg.norm((np.nanmean(m["RMFH5"], axis=1) + np.nanmean(m["RMFH1"], axis=1)) / 2 - np.nanmean(m["RCAL"], axis=1)) + np.linalg.norm((np.nanmean(m["LMFH5"], axis=1) + np.nanmean(m["LMFH1"], axis=1)) / 2 - np.nanmean(m["LCAL"], axis=1))) / 2
        foot_length_ratio = foot_length_real / foot_length_standard

        # ground position = marker on the treadmill = ankle joint center
        # TODO: Charbie -> Add the right marker name
        # ground_height = m["ground"][2]
        ground_height = 0.0
        # ankle height is the mean of all the maleolus markers
        ankle_height = ((np.nanmean(m["RSPH"][2, :]) + np.nanmean(m["RLM"][2, :]) + np.nanmean(m["LSPH"][2, :]) + np.nanmean(m["LLM"][:, 2])) / 4)
        # TODO: Charbie: Add marker_radius
        marker_radius = 0.01
        ground_pos = ankle_height - ground_height + marker_radius

        return foot_width_ratio, foot_length_ratio, ground_pos

    def _find_personalized_foot_heel(self, m):
        _, _, ground_pos = self._get_foot_characteristics(m)
        Heel_pos = np.array([0, 0, -ground_pos])
        return Heel_pos

    def _find_personalized_foot_meta1(self, m, side):
        # TODO: Charbie -> Check the contact definitions : it should not be symetric ! Find a better ref forming a scalar triangle.
        foot_width_ratio, foot_length_ratio, ground_pos = self._get_foot_characteristics(m)
        # TODO: Charbie -> Check the signs for each side
        if side == "R":
            Meta_1_pos = np.array([-0.0422882 * foot_width_ratio, 0.179793 * foot_length_ratio, -ground_pos])
        elif side == "L":
            Meta_1_pos = np.array([-0.0422882 * foot_width_ratio, 0.179793 * foot_length_ratio, -ground_pos])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")
        return Meta_1_pos

    def _find_personalized_foot_meta5(self, m, side):
        foot_width_ratio, foot_length_ratio, ground_pos = self._get_foot_characteristics(m)
        # TODO: Charbie -> Check the signs for each side
        if side == "R":
            Meta_5_pos = np.array([0.0422882 * foot_width_ratio, 0.179793 * foot_length_ratio, -ground_pos])
        elif side == "L":
            Meta_5_pos = np.array([0.0422882 * foot_width_ratio, 0.179793 * foot_length_ratio, -ground_pos])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")
        return Meta_5_pos

    def _find_personalized_gas_med_origin_position(self, m, side):
        # R_FTC - Knee joint (R_FLE + R_FME) / 2
        R_FTC = standard_model["markers"]["R_FTC"]
        R_FLE = standard_model["markers"]["R_FLE"]
        R_FME = standard_model["markers"]["R_FME"]
        knee_position_standard = (R_FLE + R_FME) / 2
        femur_length_standard = np.linalg.norm(knee_position_standard - R_FTC)
        # (LFE + MFE) / 2 - GT
        femur_length_right = np.linalg.norm((np.nanmean(m["RLFE"][2, :], axis=1)
                                            + np.nanmean(m["RMFE"][2, :], axis=1)) / 2
                                            - np.nanmean(m["RGT"][2, :], axis=1))
        femur_length_left = np.linalg.norm((np.nanmean(m["LLFE"][2, :], axis=1)
                                            + np.nanmean(m["LMFE"][2, :], axis=1)) / 2
                                            - np.nanmean(m["LGT"][2, :], axis=1))
        femur_length_real = (femur_length_right + femur_length_left) / 2

        # R_FLE - R_FME
        femur_width_standard = np.linalg.norm(R_FLE - R_FME)
        # LFE - MFE
        femur_width_right = np.linalg.norm(np.nanmean(m["RLFE"], axis=1) - np.nanmean(m["RMFE"], axis=1))
        femur_width_left = np.linalg.norm(np.nanmean(m["LLFE"], axis=1) - np.nanmean(m["LMFE"], axis=1))
        femur_width_real = (femur_width_right + femur_width_left) / 2

        if side == "R":
            gas_med_origin_standard = np.array([-0.011579, -0.35823, -0.027804])
            gas_med_ratios = np.array([femur_width_real / femur_width_standard,
                                       femur_width_real / femur_width_standard,
                                       femur_length_real / femur_length_standard])
        elif side == "L":
            gas_med_origin_standard = np.array([0.011579, -0.35823, -0.027804])
            gas_med_ratios = np.array([femur_width_real / femur_width_standard,
                                       femur_width_real / femur_width_standard,
                                       femur_length_real / femur_length_standard])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")

        return gas_med_ratios * gas_med_origin_standard

    def _find_personalized_gas_med_insertion_position(self, m, side):
        # Knee joint (R_FLE + R_FME) / 2 - Ankle joint (R_FAL + R_TAM) / 2
        R_FLE = standard_model["markers"]["R_FLE"]
        R_FME = standard_model["markers"]["R_FME"]
        R_FAL = standard_model["markers"]["R_FAL"]
        R_TAM = standard_model["markers"]["R_TAM"]
        knee_position_standard = (R_FLE + R_FME) / 2
        ankle_position_standard = (R_FAL + R_TAM) / 2
        tibia_length_standard = np.linalg.norm(knee_position_standard - ankle_position_standard)

        # (LFE + MFE) / 2 - (SPH + LM) / 2
        knee_position_right = (np.nanmean(m["RLFE"][2, :], axis=1)
                                + np.nanmean(m["RMFE"][2, :], axis=1)) / 2
        knee_position_left = (np.nanmean(m["LLFE"][2, :], axis=1)
                                + np.nanmean(m["LMFE"][2, :], axis=1)) / 2
        ankle_position_right = (np.nanmean(m["RSPH"][2, :], axis=1)
                                + np.nanmean(m["RLM"][2, :], axis=1)) / 2


        # R_FLE - R_FME
        femur_width_standard = np.linalg.norm(R_FLE - R_FME)
        gas_med_origin_standard = np.array([-0.011579, -0.35823, -0.027804])
        gas_med_origin =
        return gas_med_origin

    def _modify_kinematic_model(self):

        # TODO: Charbie: change the ranges of motion to match the article

        self.segments["Ground"] = Segment(name="Ground")

        self.segments["Pelvis"].add_range(type=Ranges.Q,
                                 min_bound=[-0.5, -0.5, 0.5, -np.pi/4, -np.pi/4, -np.pi/4],
                                 max_bound=[0.5, 0.5, 1.5, np.pi/4, np.pi/4, np.pi/4])

        self.segments["Thorax"].rotations = Rotations.NONE

        self.segments["Head"].rotations = Rotations.NONE

        # TODO: Charbie -> Verify the zero position, and modify these values if not anato
        self.segments["RHumerus"].add_range(type=Ranges.Q,
                                   min_bound=[-np.pi/4, -np.pi/4, -np.pi/4],
                                   max_bound=[np.pi/4, np.pi/4, np.pi/4])

        # TODO: Charbie -> check the axis definition for the zx choice
        self.segments["RRadius"].rotations = Rotations.ZX
        self.segments["RRadius"].add_range(type=Ranges.Q,
                                  min_bound=[-np.pi/2, -np.pi/2],
                                  max_bound=[np.pi/2, np.pi/2])

        self.segments["RHand"].rotations = Rotations.NONE

        # TODO: Charbie -> Verify the zero position, and modify these values if not anato
        self.segments["LHumerus"].add_range(type=Ranges.Q,
                                   min_bound=[-np.pi / 4, -np.pi / 4, -np.pi / 4],
                                   max_bound=[np.pi / 4, np.pi / 4, np.pi / 4])

        # TODO: Charbie -> check the axis definition for the zx choice
        self.segments["LRadius"].rotations = Rotations.ZX
        self.segments["LRadius"].add_range(type=Ranges.Q,
                                  min_bound=[-np.pi / 2, -np.pi / 2],
                                  max_bound=[np.pi / 2, np.pi / 2])

        self.segments["LHand"].rotations = Rotations.NONE

        # TODO: Charbie -> check the axis definition for the xy choice
        self.segments["RFemur"].rotations = Rotations.XY
        self.segments["RFemur"].add_range(type=Ranges.Q,
                                 min_bound=[-np.pi / 2, -np.pi / 2],
                                 max_bound=[np.pi / 2, np.pi / 2])

        # TODO: Charbie -> check the axis definition for the x choice
        self.segments["RTibia"].rotations = Rotations.X
        self.segments["RTibia"].add_range(type=Ranges.Q,
                                 min_bound=[-np.pi / 2],
                                 max_bound=[np.pi / 2])

        self.segments["RFoot"].add_range(type=Ranges.Q,
                                min_bound=[-np.pi / 4, -np.pi / 4, -np.pi / 4],
                                max_bound=[np.pi / 4, np.pi / 4, np.pi / 4])
        self.segments["RFoot"].add_contact(Contact(name="Heel_r",
                                  parent_name="RFoot",
                                  function= lambda m : self._find_personalized_foot_heel(m),
                                  axis=Translations.Z))
        self.segments["RFoot"].add_contact(Contact(name="Meta_1_r",
                                  parent_name="RFoot",
                                  function= lambda m : self._find_personalized_foot_meta1(m, side="R"),
                                  axis=Translations.Z))
        self.segments["RFoot"].add_contact(Contact(name="Meta_5_r",
                                  parent_name="RFoot",
                                  function= lambda m : self._find_personalized_foot_meta5(m, side="R"),
                                  axis=Translations.XYZ))

        # TODO: Charbie -> check the axis definition for the xy choice
        self.segments["LFemur"].rotations = Rotations.XY
        self.segments["LFemur"].add_range(type=Ranges.Q,
                                 min_bound=[-np.pi / 2, -np.pi / 2],
                                 max_bound=[np.pi / 2, np.pi / 2])

        # TODO: Charbie -> check the axis definition for the x choice
        self.segments["LTibia"].rotations = Rotations.X
        self.segments["LTibia"].add_range(type=Ranges.Q,
                                 min_bound=[-np.pi / 2],
                                 max_bound=[np.pi / 2])

        self.segments["LFoot"].add_range(type=Ranges.Q,
                                min_bound=[-np.pi / 4, -np.pi / 4, -np.pi / 4],
                                max_bound=[np.pi / 4, np.pi / 4, np.pi / 4])
        self.segments["LFoot"].add_contact(Contact(name="Heel_l",
                                  parent_name="LFoot",
                                  function= lambda m : self._find_personalized_foot_heel(m),
                                  axis=Translations.Z))
        self.segments["LFoot"].add_contact(Contact(name="Meta_1_l",
                                  parent_name="LFoot",
                                  function= lambda m : self._find_personalized_foot_meta1(m, side="L"),
                                  axis=Translations.Z))
        self.segments["LFoot"].add_contact(Contact(name="Meta_5_l",
                                  parent_name="LFoot",
                                  function= lambda m : self._find_personalized_foot_meta5(m, side="L"),
                                  axis=Translations.XYZ))


    @property
    def dof_index(self) -> dict[str, tuple[int, ...]]:
        """
        Returns a dictionary with all the dof to export to the C3D and their corresponding XYZ values in the generalized
        coordinate vector
        """

        # TODO: Charbie -> check these
        return {"LHip": (36, 37, 38),
                "LKnee": (39, 40, 41),
                "LAnkle": (42, 43, 44),
                "LAbsAnkle": (42, 43, 44),
                "LFootProgress": (42, 43, 44),
                "RHip": (27, 28, 29),
                "RKnee": (30, 31, 32),
                "RAnkle": (33, 34, 35),
                "RAbsAnkle": (33, 34, 35),
                "RFootProgress": (33, 34, 35),
                "LShoulder": (18, 19, 20),
                "LElbow": (21, 22, 23),
                "LWrist": (24, 25, 26),
                "RShoulder": (9, 10, 11),
                "RElbow": (12, 13, 14),
                "RWrist": (15, 16, 17),
                "LNeck": None,
                "RNeck": None,
                "LSpine": None,
                "RSpine": None,
                "LHead": None,
                "RHead": None,
                "LThorax": (6, 7, 8),
                "RThorax": (6, 7, 8),
                "LPelvis": (3, 4, 5),
                "RPelvis": (3, 4, 5),
                }

    def _define_muscle_model(self):
        # Muscle groups
        self.muscle_groups["RFemur_Foot"] = MuscleGroup(name="RFemur_Foot",
                                                         origin_parent_name="RFemur",
                                                         insertion_parent_name="RFoot")
        self.muscle_groups["RTibia_Foot"] = MuscleGroup(name="RTibia_Foot",
                                                         origin_parent_name="RTibia",
                                                         insertion_parent_name="RFoot")
        self.muscle_groups["LFemur_Foot"] = MuscleGroup(name="LFemur_Foot",
                                                         origin_parent_name="LFemur",
                                                         insertion_parent_name="LFoot")
        self.muscle_groups["LTibia_Foot"] = MuscleGroup(name="RTibia_Foot",
                                                         origin_parent_name="LTibia",
                                                         insertion_parent_name="LFoot")

        # Muscles
        self.muscles["RGas_med"] = Muscle(name="RGas_med",
                                            type=MuscleType.HILLTHELEN,
                                            state_type=MuscleStateType.DEGROOTE,
                                            muscle_group="RFemur_Foot",
                                            origin_position_function=lambda m: self._find_personalized_gas_med_origin_position(m, side="R"),
                                            insertion_position_function=lambda m: self._find_personalized_gas_med_insertion_position(m, side="R"),
                                            optimal_length_function=lambda m: self._find_personalized_gas_med_optimal_length(m, side="R"),
                                            maximal_force_function=100,  # Hard coded since this value should really not be reached during walking
                                            tendon_slack_length_function=lambda m: self._find_personalized_gas_med_tendon_slack_length(m, side="R"),
                                            pennation_angle_function=lambda m: self._find_personalized_gas_med_pennation_angle(m, side="R")
                                          )
