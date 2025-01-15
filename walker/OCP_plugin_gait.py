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
    ViaPoint,
    MeshFile,
)
import numpy as np
import biorbd
import pandas as pd

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
        name: str,
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
        name
            The name of the model/participant
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
        super(OCPPluginGait, self).__init__(name,
                                            body_mass,
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
        foot_width_standard, foot_width_real = self._foot_width(m)
        foot_length_standard, foot_length_real = self._foot_length(m)

        foot_width_ratio = foot_width_real / foot_width_standard
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
            Meta_1_pos = np.array([-0.0422882 * foot_width_ratio, -0.179793 * foot_length_ratio, -ground_pos])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")
        return Meta_1_pos

    def _find_personalized_foot_meta5(self, m, side):
        foot_width_ratio, foot_length_ratio, ground_pos = self._get_foot_characteristics(m)
        # TODO: Charbie -> Check the signs for each side
        if side == "R":
            Meta_5_pos = np.array([0.0422882 * foot_width_ratio, 0.179793 * foot_length_ratio, -ground_pos])
        elif side == "L":
            Meta_5_pos = np.array([0.0422882 * foot_width_ratio, -0.179793 * foot_length_ratio, -ground_pos])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")
        return Meta_5_pos

    def _knee_width(self, m):
        R_FLE = standard_model["markers"]["R_FLE"]
        R_FME = standard_model["markers"]["R_FME"]
        # R_FLE - R_FME
        knee_width_standard = np.linalg.norm(R_FLE - R_FME)
        # LFE - MFE
        knee_width_right = np.linalg.norm(np.nanmean(m["RLFE"], axis=1) - np.nanmean(m["RMFE"], axis=1))
        knee_width_left = np.linalg.norm(np.nanmean(m["LLFE"], axis=1) - np.nanmean(m["LMFE"], axis=1))
        knee_width_real = (knee_width_right + knee_width_left) / 2
        return knee_width_standard, knee_width_real

    def _ankle_width(self, m):
        R_FAL = standard_model["markers"]["R_FAL"]
        R_TAM = standard_model["markers"]["R_TAM"]
        # R_FAL - R_TAM
        ankle_width_standard = np.linalg.norm(R_FAL - R_TAM)
        # SPH - LM
        ankle_width_right = np.linalg.norm(np.nanmean(m["RSPH"], axis=1) - np.nanmean(m["RLM"], axis=1))
        ankle_width_left = np.linalg.norm(np.nanmean(m["LSPH"], axis=1) - np.nanmean(m["LLM"], axis=1))
        ankle_width_real = (ankle_width_right + ankle_width_left) / 2
        return ankle_width_standard, ankle_width_real

    def _femur_length(self, m):
        # R_FTC - Knee joint (R_FLE + R_FME) / 2
        R_FTC = standard_model["markers"]["R_FTC"]
        R_FLE = standard_model["markers"]["R_FLE"]
        R_FME = standard_model["markers"]["R_FME"]
        knee_position_standard = (R_FLE + R_FME) / 2
        femur_length_standard = np.linalg.norm(knee_position_standard - R_FTC)
        # (LFE + MFE) / 2 - GT
        femur_length_right = np.linalg.norm((np.nanmean(m["RLFE"][:3, :], axis=1)
                                            + np.nanmean(m["RMFE"][:3, :], axis=1)) / 2
                                            - np.nanmean(m["RGT"][:3, :], axis=1))
        femur_length_left = np.linalg.norm((np.nanmean(m["LLFE"][:3, :], axis=1)
                                            + np.nanmean(m["LMFE"][:3, :], axis=1)) / 2
                                            - np.nanmean(m["LGT"][:3, :], axis=1))
        femur_length_real = (femur_length_right + femur_length_left) / 2
        return femur_length_standard, femur_length_real


    def _tibia_length(self, m):
        # Knee joint (R_FLE + R_FME) / 2 - Ankle joint (R_FAL + R_TAM) / 2
        R_FLE = standard_model["markers"]["R_FLE"]
        R_FME = standard_model["markers"]["R_FME"]
        R_FAL = standard_model["markers"]["R_FAL"]
        R_TAM = standard_model["markers"]["R_TAM"]
        knee_position_standard = (R_FLE + R_FME) / 2
        ankle_position_standard = (R_FAL + R_TAM) / 2
        tibia_length_standard = np.linalg.norm(knee_position_standard - ankle_position_standard)

        # (LFE + MFE) / 2 - (SPH + LM) / 2
        knee_position_right = (np.nanmean(m["RLFE"][:3, :], axis=1)
                                + np.nanmean(m["RMFE"][:3, :], axis=1)) / 2
        knee_position_left = (np.nanmean(m["LLFE"][:3, :], axis=1)
                                + np.nanmean(m["LMFE"][:3, :], axis=1)) / 2
        ankle_position_right = (np.nanmean(m["RSPH"][:3, :], axis=1)
                                + np.nanmean(m["RLM"][:3, :], axis=1)) / 2
        ankle_position_left = (np.nanmean(m["LSPH"][:3, :], axis=1)
                                + np.nanmean(m["LLM"][:3, :], axis=1)) / 2
        tibia_length_real = (np.linalg.norm(knee_position_right - ankle_position_right)
                            + np.linalg.norm(knee_position_left - ankle_position_left)) / 2
        return tibia_length_standard, tibia_length_real


    def _foot_length(self, m):
        # TODO: Charbie -> Verify that TT2 and FMP2 are placed at the same position (otherwise M5 - M1)
        # Toes (R_FM2) - Heel (FCC)
        R_FM2 = standard_model["markers"]["R_FM2"]
        FCC = standard_model["markers"]["R_FCC"]
        foot_length_standard = np.linalg.norm(R_FM2 - FCC)
        # TT2 - CAL
        foot_length_right = np.linalg.norm(np.nanmean(m["RTT2"][:3, :], axis=1)
                               - np.nanmean(m["RCAL"][:3, :], axis=1))
        foot_length_left = np.linalg.norm(np.nanmean(m["LTT2"][:3, :], axis=1)
                               - np.nanmean(m["LCAL"][:3, :], axis=1))
        foot_length_real = (foot_length_right + foot_length_left) / 2
        return foot_length_standard, foot_length_real


    def _foot_width(self, m):
        # TODO: Charbie -> Verify if shoes are used of barefoot
        # R_FM5 - R_FM1
        R_FM5 = standard_model["markers"]["R_FM5"]
        R_FM1 = standard_model["markers"]["R_FM1"]
        foot_width_standard = np.linalg.norm(R_FM5 - R_FM1)

        # MFH5 - MFH1
        foot_width_right = np.linalg.norm(np.nanmean(m["RMFH5"][:3, :], axis=1)
                               - np.nanmean(m["RMFH1"][:3, :], axis=1))
        foot_width_left = np.linalg.norm(np.nanmean(m["LMFH5"][:3, :], axis=1)
                               - np.nanmean(m["LMFH5"][:3, :], axis=1))
        foot_width_real = (foot_width_right + foot_width_left) / 2
        return foot_width_standard, foot_width_real


    def _find_personalized_gas_med_origin_position(self, m, side):
        knee_width_standard, knee_width_real = self._knee_width(m)
        femur_length_standard, femur_length_real = self._femur_length(m)

        if side == "R":
            gas_med_origin_standard = np.array([-0.011579, -0.35823, -0.027804])
        elif side == "L":
            gas_med_origin_standard = np.array([-0.011579, 0.35823, -0.027804])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")
        gas_med_ratios = np.array([knee_width_real / knee_width_standard,
                                   knee_width_real / knee_width_standard,
                                   femur_length_real / femur_length_standard])

        return gas_med_ratios * gas_med_origin_standard

    def _find_personalized_gas_med_insertion_position(self, m, side):
        ankle_width_standard, ankle_width_real = self._ankle_width(m)
        tibia_length_standard, tibia_length_real = self._tibia_length(m)

        if side == "R":
            gas_med_insertion_standard = np.array([0.0033495, 0.03194, -0.0063202])
        elif side == "L":
            gas_med_insertion_standard = np.array([0.0033495, -0.03194, -0.0063202])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")
        gas_med_ratios = np.array([ankle_width_real / ankle_width_standard,
                                      ankle_width_real / ankle_width_standard,
                                      tibia_length_real / tibia_length_standard])

        return gas_med_ratios * gas_med_insertion_standard

    def _find_personalized_gas_med_via_point_position(self, m, side):
        knee_width_standard, knee_width_real = self._knee_width(m)
        ankle_width_standard, ankle_width_real = self._ankle_width(m)
        tibia_length_standard, tibia_length_real = self._tibia_length(m)

        if side == "R":
            gas_med_via_point_standard = np.array([-0.021594, -0.048462, -0.029356])
        elif side == "L":
            gas_med_via_point_standard = np.array([-0.021594, 0.048462, -0.029356])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")
        gas_med_via_point_ratios = np.array([(ankle_width_real / ankle_width_standard + knee_width_real / knee_width_standard) / 2,
                                   (ankle_width_real / ankle_width_standard + knee_width_real / knee_width_standard) / 2,
                                   tibia_length_real / tibia_length_standard])

        return gas_med_via_point_ratios * gas_med_via_point_standard


    def _find_personalized_gas_med_tendon_slack_length(self, model, m):
        csv = pd.read_csv(f"data/Sujet_{self.name}.csv")
        MTJ_index = np.where(np.array(csv)[:, 0] == "MTJ")[0][0]
        gas_med_tendon_slack_length = float(np.array(csv)[MTJ_index, 1]) * 0.01
        raise RuntimeError("Please see the dev ;p")


    def _find_personalized_gas_med_optimal_length(self, model, m):
        origin_position_in_femur_right = self._find_personalized_gas_med_origin_position(m, "R")
        scs_femur_right = model.segments["RFemur"].segment_coordinate_system.transpose.mean_scs
        origin_position_in_global_right = scs_femur_right @ np.hstack((origin_position_in_femur_right, 1))

        insertion_position_in_foot_right = self._find_personalized_gas_med_insertion_position(m, "R")
        scs_foot_right = model.segments["RFoot"].segment_coordinate_system.transpose.mean_scs
        insertion_position_in_global_right = scs_foot_right @ np.hstack((insertion_position_in_foot_right, 1))

        via_point_position_in_tibia_right = self._find_personalized_gas_med_via_point_position(m, "R")
        scs_tibia_right = model.segments["RTibia"].segment_coordinate_system.transpose.mean_scs
        via_point_position_in_global_right = scs_tibia_right @ np.hstack((via_point_position_in_tibia_right, 1))

        length_right = np.linalg.norm(origin_position_in_global_right[:3] - via_point_position_in_global_right[:3]) + np.linalg.norm(via_point_position_in_global_right[:3] - insertion_position_in_global_right[:3])
        length_standard = np.linalg.norm(standard_model["markers"]["gas_med_r_origin"] - standard_model["markers"]["gas_med_r_via_point"]) + np.linalg.norm(standard_model["markers"]["gas_med_r_via_point"] - standard_model["markers"]["gas_med_r_insertion"])
        optimal_length_standard = 0.044721

        return length_right / length_standard * optimal_length_standard


    def _find_personalized_gas_med_pennation_angle(self, model, m, side):
        raise RuntimeError("Please see the dev ;p")


    def _find_personalized_gas_lat_origin_position(self, m, side):
        knee_width_standard, knee_width_real = self._knee_width(m)
        femur_length_standard, femur_length_real = self._femur_length(m)

        if side == "R":
            gas_lat_origin_standard = np.array([-0.014132, -0.35978, 0.032181])
        elif side == "L":
            gas_lat_origin_standard = np.array([-0.014132, 0.35978, 0.032181])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")
        gas_lat_ratios = np.array([knee_width_real / knee_width_standard,
                                   knee_width_real / knee_width_standard,
                                   femur_length_real / femur_length_standard])

        return gas_lat_ratios * gas_lat_origin_standard

    def _find_personalized_gas_lat_insertion_position(self, m, side):
        ankle_width_standard, ankle_width_real = self._ankle_width(m)
        tibia_length_standard, tibia_length_real = self._tibia_length(m)

        if side == "R":
            gas_lat_insertion_standard = np.array([0.0033495, 0.03194, -0.0063202])
        elif side == "L":
            gas_lat_insertion_standard = np.array([0.0033495, -0.03194, -0.0063202])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")
        gas_lat_ratios = np.array([ankle_width_real / ankle_width_standard,
                                      ankle_width_real / ankle_width_standard,
                                      tibia_length_real / tibia_length_standard])

        return gas_lat_ratios * gas_lat_insertion_standard

    def _find_personalized_gas_lat_via_point_position(self, m, side):
        knee_width_standard, knee_width_real = self._knee_width(m)
        ankle_width_standard, ankle_width_real = self._ankle_width(m)
        tibia_length_standard, tibia_length_real = self._tibia_length(m)

        if side == "R":
            gas_lat_via_point_standard = np.array([-0.024082, -0.047865, 0.023385])
        elif side == "L":
            gas_lat_via_point_standard = np.array([-0.024082, 0.047865, 0.023385])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")
        gas_lat_via_point_ratios = np.array([(ankle_width_real / ankle_width_standard + knee_width_real / knee_width_standard) / 2,
                                   (ankle_width_real / ankle_width_standard + knee_width_real / knee_width_standard) / 2,
                                   tibia_length_real / tibia_length_standard])

        return gas_lat_via_point_ratios * gas_lat_via_point_standard


    def _find_personalized_gas_lat_optimal_length(self, model, m):
        origin_position_in_femur_right = self._find_personalized_gas_lat_origin_position(m, "R")
        scs_femur_right = model.segments["RFemur"].segment_coordinate_system.transpose.mean_scs
        origin_position_in_global_right = scs_femur_right @ np.hstack((origin_position_in_femur_right, 1))

        insertion_position_in_foot_right = self._find_personalized_gas_lat_insertion_position(m, "R")
        scs_foot_right = model.segments["RFoot"].segment_coordinate_system.transpose.mean_scs
        insertion_position_in_global_right = scs_foot_right @ np.hstack((insertion_position_in_foot_right, 1))

        via_point_position_in_tibia_right = self._find_personalized_gas_lat_via_point_position(m, "R")
        scs_tibia_right = model.segments["RTibia"].segment_coordinate_system.transpose.mean_scs
        via_point_position_in_global_right = scs_tibia_right @ np.hstack((via_point_position_in_tibia_right, 1))

        length_right = np.linalg.norm(origin_position_in_global_right[:3] - via_point_position_in_global_right[:3]) + np.linalg.norm(via_point_position_in_global_right[:3] - insertion_position_in_global_right[:3])
        length_standard = np.linalg.norm(standard_model["markers"]["gas_lat_r_origin"] - standard_model["markers"]["gas_lat_r_via_point"]) + np.linalg.norm(standard_model["markers"]["gas_lat_r_via_point"] - standard_model["markers"]["gas_lat_r_insertion"])
        optimal_length_standard = 0.06375

        return length_right / length_standard * optimal_length_standard


    def _find_personalized_soleus_origin_position(self, m, side):
        knee_width_standard, knee_width_real = self._knee_width(m)
        ankle_width_standard, ankle_width_real = self._ankle_width(m)
        width_standard = (knee_width_standard + ankle_width_standard) / 2
        width_real = (knee_width_real + ankle_width_real) / 2
        tibia_length_standard, tibia_length_real = self._tibia_length(m)

        if side == "R":
            soleus_origin_standard = np.array([-0.0023883, -0.15255, 0.0070653])
        elif side == "L":
            soleus_origin_standard = np.array([-0.0023883, 0.15255, 0.0070653])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")
        soleus_ratios = np.array([width_real / width_standard,
                                   width_real / width_standard,
                                   tibia_length_real / tibia_length_standard])

        return soleus_ratios * soleus_origin_standard

    def _find_personalized_soleus_insertion_position(self, m, side):
        foot_width_standard, foot_width_real = self._foot_width(m)
        foot_length_standard, foot_length_real = self._foot_length(m)
        cal_height_real = (np.nanmean(m["RCAL"], axis=1)[2] + np.nanmean(m["LCAL"], axis=1)[2]) / 2

        if side == "R":
            soleus_insertion_standard = np.array([0.0033495, 0.03194])
        elif side == "L":
            soleus_insertion_standard = np.array([0.0033495, -0.03194])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")
        soleus_ratios = np.array([foot_length_real / foot_length_standard,
                                      foot_width_real / foot_width_standard])

        return np.hstack((soleus_ratios * soleus_insertion_standard, cal_height_real))


    def _find_personalized_soleus_optimal_length(self, model, m):
        origin_position_in_tibia_right = self._find_personalized_soleus_origin_position(m, "R")
        scs_tibia_right = model.segments["RTibia"].segment_coordinate_system.transpose.mean_scs
        origin_position_in_global_right = scs_tibia_right @ np.hstack((origin_position_in_tibia_right, 1))

        insertion_position_in_foot_right = self._find_personalized_soleus_insertion_position(m, "R")
        scs_foot_right = model.segments["RFoot"].segment_coordinate_system.transpose.mean_scs
        insertion_position_in_global_right = scs_foot_right @ np.hstack((insertion_position_in_foot_right, 1))

        length_right = np.linalg.norm(origin_position_in_global_right[:3] - insertion_position_in_global_right[:3])
        length_standard = np.linalg.norm(standard_model["markers"]["soleus_r_origin"] - standard_model["markers"]["soleus_r_insertion"])
        optimal_length_standard = 0.029856

        return length_right / length_standard * optimal_length_standard


    def _find_personalized_tib_ant_origin_position(self, m, side):
        knee_width_standard, knee_width_real = self._knee_width(m)
        ankle_width_standard, ankle_width_real = self._ankle_width(m)
        width_standard = (knee_width_standard + ankle_width_standard) / 2
        width_real = (knee_width_real + ankle_width_real) / 2
        tibia_length_standard, tibia_length_real = self._tibia_length(m)

        if side == "R":
            tib_ant_origin_standard = np.array([0.017812, -0.16161, 0.011444])
        elif side == "L":
            tib_ant_origin_standard = np.array([0.017812, 0.16161, 0.011444])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")
        tib_ant_ratios = np.array([width_real / width_standard,
                                   width_real / width_standard,
                                   tibia_length_real / tibia_length_standard])

        return tib_ant_ratios * tib_ant_origin_standard


    def _find_personalized_tib_ant_insertion_position(self, m, side):
        foot_width_standard, foot_width_real = self._foot_width(m)
        foot_length_standard, foot_length_real = self._foot_length(m)
        cal_height_real = (np.nanmean(m["RCAL"], axis=1)[2] + np.nanmean(m["LCAL"], axis=1)[2]) / 2

        if side == "R":
            tib_ant_insertion_standard = np.array([0.11624, 0.017744])
        elif side == "L":
            tib_ant_insertion_standard = np.array([0.11624, -0.017744])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")
        tib_ant_ratios = np.array([foot_length_real / foot_length_standard,
                                      foot_width_real / foot_width_standard])

        return np.hstack((tib_ant_ratios * tib_ant_insertion_standard, cal_height_real))


    def _find_personalized_tib_ant_via_point_position(self, m, side):
        knee_width_standard, knee_width_real = self._knee_width(m)
        ankle_width_standard, ankle_width_real = self._ankle_width(m)
        tibia_length_standard, tibia_length_real = self._tibia_length(m)

        if side == "R":
            tib_ant_via_point_standard = np.array([0.032739, -0.39317, 0.017613])
        elif side == "L":
            tib_ant_via_point_standard = np.array([0.032739, 0.39317, 0.017613])
        else:
            raise RuntimeError("The side should be either 'R' or 'L'")
        tib_ant_via_point_ratios = np.array([(ankle_width_real / ankle_width_standard + knee_width_real / knee_width_standard) / 2,
                                   (ankle_width_real / ankle_width_standard + knee_width_real / knee_width_standard) / 2,
                                   tibia_length_real / tibia_length_standard])

        return tib_ant_via_point_ratios * tib_ant_via_point_standard


    def _find_personalized_tib_ant_optimal_length(self, model, m):
        origin_position_in_tibia_right = self._find_personalized_tib_ant_origin_position(m, "R")
        scs_tibia_right = model.segments["RTibia"].segment_coordinate_system.transpose.mean_scs
        origin_position_in_global_right = scs_tibia_right @ np.hstack((origin_position_in_tibia_right, 1))

        via_point_position_in_tibia_right = self._find_personalized_gas_lat_via_point_position(m, "R")
        scs_tibia_right = model.segments["RTibia"].segment_coordinate_system.transpose.mean_scs
        via_point_position_in_global_right = scs_tibia_right @ np.hstack((via_point_position_in_tibia_right, 1))

        insertion_position_in_foot_right = self._find_personalized_soleus_insertion_position(m, "R")
        scs_foot_right = model.segments["RFoot"].segment_coordinate_system.transpose.mean_scs
        insertion_position_in_global_right = scs_foot_right @ np.hstack((insertion_position_in_foot_right, 1))

        length_right = np.linalg.norm(origin_position_in_global_right[:3] - via_point_position_in_global_right[:3]) + np.linalg.norm(via_point_position_in_global_right[:3] - insertion_position_in_global_right[:3])
        length_standard = np.linalg.norm(standard_model["markers"]["tib_ant_r_origin"] - standard_model["markers"]["tib_ant_r_via_point"]) + np.linalg.norm(standard_model["markers"]["tib_ant_r_via_point"] - standard_model["markers"]["soleus_r_insertion"])
        optimal_length_standard = 0.097554

        return length_right / length_standard * optimal_length_standard


    def _modify_kinematic_model(self):

        # TODO: Charbie: change the ranges of motion to match the article

        self.segments["Ground"] = Segment(name="Ground")
        # self.segments["Ground"].mesh_file = MeshFile(mesh_file_name="mesh/treadmill.vtp",
        #                                             scaling_function=lambda m: np.array([1, 1, 1]),
        #                                             rotation_function=lambda m: np.array([0, 0, 0]),
        #                                             translation_function=lambda m: np.array([0, 0, 0]))
        # self.segments["Ground"].mesh = None

        self.segments["Pelvis"].add_range(range_type=Ranges.Q,
                                 min_bound=[-0.5, -0.5, 0.5, -np.pi/4, -np.pi/4, -np.pi/4],
                                 max_bound=[0.5, 0.5, 1.5, np.pi/4, np.pi/4, np.pi/4])
        self.segments["Pelvis"].mesh_file = MeshFile(mesh_file_name="mesh/pelvis.vtp",
                                                    scaling_function=lambda m: np.array([1, 1, 1]),
                                                    rotation_function=lambda m: np.array([0, 0, 0]),
                                                    translation_function=lambda m: np.array([0, 0, 0]))
        self.segments["Pelvis"].mesh = None


        self.segments["Thorax"].rotations = Rotations.NONE

        self.segments["Head"].rotations = Rotations.NONE

        # TODO: Charbie -> Verify the zero position, and modify these values if not anato
        self.segments["RHumerus"].add_range(range_type=Ranges.Q,
                                   min_bound=[-np.pi/4, -np.pi/4, -np.pi/4],
                                   max_bound=[np.pi/4, np.pi/4, np.pi/4])

        # TODO: Charbie -> check the axis definition for the zx choice
        self.segments["RRadius"].rotations = Rotations.ZX
        self.segments["RRadius"].add_range(range_type=Ranges.Q,
                                  min_bound=[-np.pi/2, -np.pi/2],
                                  max_bound=[np.pi/2, np.pi/2])

        self.segments["RHand"].rotations = Rotations.NONE

        # TODO: Charbie -> Verify the zero position, and modify these values if not anato
        self.segments["LHumerus"].add_range(range_type=Ranges.Q,
                                   min_bound=[-np.pi / 4, -np.pi / 4, -np.pi / 4],
                                   max_bound=[np.pi / 4, np.pi / 4, np.pi / 4])

        # TODO: Charbie -> check the axis definition for the zx choice
        self.segments["LRadius"].rotations = Rotations.ZX
        self.segments["LRadius"].add_range(range_type=Ranges.Q,
                                  min_bound=[-np.pi / 2, -np.pi / 2],
                                  max_bound=[np.pi / 2, np.pi / 2])

        self.segments["LHand"].rotations = Rotations.NONE

        # TODO: Charbie -> check the axis definition for the xy choice
        self.segments["RFemur"].rotations = Rotations.XY
        self.segments["RFemur"].add_range(range_type=Ranges.Q,
                                 min_bound=[-np.pi / 2, -np.pi / 2],
                                 max_bound=[np.pi / 2, np.pi / 2])

        # TODO: Charbie -> check the axis definition for the x choice
        self.segments["RTibia"].rotations = Rotations.X
        self.segments["RTibia"].add_range(range_type=Ranges.Q,
                                 min_bound=[-np.pi / 2],
                                 max_bound=[np.pi / 2])

        self.segments["RFoot"].add_range(range_type=Ranges.Q,
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
        self.segments["LFemur"].add_range(range_type=Ranges.Q,
                                 min_bound=[-np.pi / 2, -np.pi / 2],
                                 max_bound=[np.pi / 2, np.pi / 2])

        # TODO: Charbie -> check the axis definition for the x choice
        self.segments["LTibia"].rotations = Rotations.X
        self.segments["LTibia"].add_range(range_type=Ranges.Q,
                                 min_bound=[-np.pi / 2],
                                 max_bound=[np.pi / 2])

        self.segments["LFoot"].add_range(range_type=Ranges.Q,
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
        self.muscle_groups["LTibia_Foot"] = MuscleGroup(name="LTibia_Foot",
                                                         origin_parent_name="LTibia",
                                                         insertion_parent_name="LFoot")

        # Muscles
        self.muscles["RGas_med"] = Muscle(name="RGas_med",
                                            muscle_type=MuscleType.HILLTHELEN,
                                            state_type=MuscleStateType.DEGROOTE,
                                            muscle_group="RFemur_Foot",
                                            origin_position_function=lambda m: self._find_personalized_gas_med_origin_position(m, side="R"),
                                            insertion_position_function=lambda m: self._find_personalized_gas_med_insertion_position(m, side="R"),
                                            optimal_length_function=lambda model, m: self._find_personalized_gas_med_optimal_length(model, m),
                                            maximal_force_function=lambda m: 1600.0,  # Hard coded since this value should really not be reached during walking (but still taken from Moissenet et al. 2019)
                                            tendon_slack_length_function=lambda model, m: 0.40547,  # self._find_personalized_gas_med_tendon_slack_length(model, m),
                                            pennation_angle_function=lambda model, m: 0.29671,  # self._find_personalized_gas_med_pennation_angle(model, m, side="R")
                                          )
        self.via_points["gas_med_r"] = ViaPoint(name="gas_med_r",
                                                position_function=lambda m: self._find_personalized_gas_med_via_point_position(m, side="R"),
                                                parent_name="RTibia",
                                                muscle_name="RGas_med",
                                                muscle_group="RFemur_Foot")

        self.muscles["RGas_lat"] = Muscle(name="RGas_lat",
                                            muscle_type=MuscleType.HILLTHELEN,
                                            state_type=MuscleStateType.DEGROOTE,
                                            muscle_group="RFemur_Foot",
                                            origin_position_function=lambda m: self._find_personalized_gas_lat_origin_position(m, side="R"),
                                            insertion_position_function=lambda m: self._find_personalized_gas_lat_insertion_position(m, side="R"),
                                            optimal_length_function=lambda model, m: self._find_personalized_gas_lat_optimal_length(model, m),
                                            maximal_force_function=lambda m: 700.0,  # Hard coded since this value should really not be reached during walking (but still taken from Moissenet et al. 2019)
                                            tendon_slack_length_function=lambda model, m: 0.38349,
                                            pennation_angle_function=lambda model, m: 0.13963,
                                          )
        self.via_points["gas_lat_r"] = ViaPoint(name="gas_lat_r",
                                                position_function=lambda m: self._find_personalized_gas_lat_via_point_position(m, side="R"),
                                                parent_name="RTibia",
                                                muscle_name="RGas_lat",
                                                muscle_group="RFemur_Foot")

        self.muscles["RSoleus"] = Muscle(name="RSoleus",
                                            muscle_type=MuscleType.HILLTHELEN,
                                            state_type=MuscleStateType.DEGROOTE,
                                            muscle_group="RTibia_Foot",
                                            origin_position_function=lambda m: self._find_personalized_soleus_origin_position(m, side="R"),
                                            insertion_position_function=lambda m: self._find_personalized_soleus_insertion_position(m, side="R"),
                                            optimal_length_function=lambda model, m: self._find_personalized_soleus_optimal_length(model, m),
                                            maximal_force_function=lambda m: 4000.0,  # Hard coded since this value should really not be reached during walking (but still taken from Moissenet et al. 2019)
                                            tendon_slack_length_function=lambda model, m: 0.26672,  # TODO: Charbie
                                            pennation_angle_function=lambda model, m: 0.43633,
                                          )

        self.muscles["RTib_ant"] = Muscle(name="RTib_ant",
                                            muscle_type=MuscleType.HILLTHELEN,
                                            state_type=MuscleStateType.DEGROOTE,
                                            muscle_group="RTibia_Foot",
                                            origin_position_function=lambda m: self._find_personalized_tib_ant_origin_position(m, side="R"),
                                            insertion_position_function=lambda m: self._find_personalized_tib_ant_insertion_position(m, side="R"),
                                            optimal_length_function=lambda model, m: self._find_personalized_tib_ant_optimal_length(model, m),
                                            maximal_force_function=lambda m: 3000.0,  # Hard coded since this value should really not be reached during walking (but still taken from Moissenet et al. 2019)
                                            tendon_slack_length_function=lambda model, m: 0.22199,
                                            pennation_angle_function=lambda model, m: 0.087266,
                                          )
        self.via_points["tib_ant_r"] = ViaPoint(name="tib_ant_r",
                                                position_function=lambda m: self._find_personalized_tib_ant_via_point_position(m, side="R"),
                                                parent_name="RTibia",
                                                muscle_name="RTib_ant",
                                                muscle_group="RTibia_Foot")