from biorbd.model_creation import (
    Axis,
    BiomechanicalModel,
    BiomechanicalModelReal,
    SegmentCoordinateSystem,
    InertiaParameters,
    Mesh,
    Segment,
    Marker,
    Contact,
    Translations,
    Rotations,
    Ranges,
    MuscleGroup,
    Muscle,
    MuscleType,
)
import numpy as np
import biorbd


def chord_function(offset, known_center_of_rotation, center_of_rotation_marker, plane_marker, direction: int = 1):
    n_frames = offset.shape[0]

    # Create a coordinate system from the markers
    axis1 = plane_marker[:3, :] - known_center_of_rotation[:3, :]
    axis2 = center_of_rotation_marker[:3, :] - known_center_of_rotation[:3, :]
    axis3 = np.cross(axis1, axis2, axis=0)
    axis1 = np.cross(axis2, axis3, axis=0)
    axis1 /= np.linalg.norm(axis1, axis=0)
    axis2 /= np.linalg.norm(axis2, axis=0)
    axis3 /= np.linalg.norm(axis3, axis=0)
    rt = np.identity(4)
    rt = np.repeat(rt, n_frames, axis=1).reshape((4, 4, n_frames))
    rt[:3, 0, :] = axis1
    rt[:3, 1, :] = axis2
    rt[:3, 2, :] = axis3
    rt[:3, 3, :] = known_center_of_rotation[:3, :]

    # The point of interest is the chord from center_of_rotation_marker that has length 'offset' assuming
    # the diameter is the distance between center_of_rotation_marker and known_center_of_rotation.
    # To compute this, project in the rt knowing that by construction, known_center_of_rotation is at 0, 0, 0
    # and center_of_rotation_marker is at a diameter length on y
    diameter = np.linalg.norm(known_center_of_rotation[:3, :] - center_of_rotation_marker[:3, :], axis=0)
    x = offset * direction * np.sqrt(diameter**2 - offset**2) / diameter
    y = (diameter**2 - offset**2) / diameter

    # project the computed point in the global reference frame
    vect = np.concatenate((x[np.newaxis, :], y[np.newaxis, :], np.zeros((1, n_frames)), np.ones((1, n_frames))))

    def rt_times_vect(m1, m2):
        return np.einsum("ijk,jk->ik", m1, m2)

    return rt_times_vect(rt, vect)


def point_on_vector(coef: float, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """
    Computes the 3d position of a point using this equation: start + coef * (end - start)

    Parameters
    ----------
    coef
        The coefficient of the length of the segment to use. It is given from the starting point
    start
        The starting point of the segment
    end
        The end point of the segment

    Returns
    -------
    The 3d position of the point
    """

    return start + coef * (end - start)


def project_point_on_line(start_line: np.ndarray, end_line: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Project a point on a line defined by to points (start_line and end_line)

    Parameters
    ----------
    start_line
        The starting point of the line
    end_line
        The ending point of the line
    point
        The point to project

    Returns
    -------
    The projected point
    -------

    """

    def dot(v1, v2):
        return np.einsum("ij,ij->j", v1, v2)

    sp = (point - start_line)[:3, :]
    line = (end_line - start_line)[:3, :]
    return start_line[:3, :] + dot(sp, line) / dot(line, line) * line


class SimplePluginGait(BiomechanicalModel):
    """
    This is the implementation of the Plugin Gait (from Plug-in Gait Reference Guide
    https://docs.vicon.com/display/Nexus212/PDF+downloads+for+Vicon+Nexus)
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
        super(SimplePluginGait, self).__init__()
        self.name = name
        self.body_mass = body_mass
        self.include_upper_body = include_upper_body
        self.shoulder_offset = shoulder_offset
        self.elbow_width = elbow_width
        self.wrist_width = wrist_width
        self.hand_thickness = hand_thickness
        self.leg_length = leg_length
        self.ankle_width = ankle_width

        self._define_kinematic_model()

    def _define_kinematic_model(self):
        # Pelvis: verified, The radii of gyration were computed using InterHip normalisation
        # Thorax: verified
        # Head: verified
        # Humerus: verified
        # Radius: verified
        # Hand: Moved the hand joint center to WJC
        # Femur: verified
        # Knee: Used mid-point of 'KNM' and 'KNE' as KJC
        # Ankle: As for knee, we have access to a much easier medial marker (ANKM), so it was used instead
        self.segments["Ground"] = Segment(name="Ground")

        self.segments["Pelvis"] = Segment(
            name="Pelvis",
            parent_name="Ground",
            translations=Translations.XYZ,
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=self._pelvis_joint_center,
                first_axis=Axis(name=Axis.Name.X, start=lambda m, bio: (m["LPSIS"] + m["RPSIS"]) / 2, end="RASIS"),
                second_axis=Axis(name=Axis.Name.Y, start="RASIS", end="LASIS"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("LPSIS", "RPSIS", "RASIS", "LASIS", "LPSIS")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.145 * self.body_mass,
                center_of_mass=self._pelvis_center_of_mass,
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.145 * self.body_mass,
                    coef=(0.31, 0.31, 0.3),
                    start=self._pelvis_joint_center(m, bio),
                    end=self._pelvis_center_of_mass(m, bio),
                ),
            ),
        )
        # self.add_marker("Pelvis", "SACR", is_technical=True, is_anatomical=True)
        self.segments["Pelvis"].add_marker(Marker("LPSIS", is_technical=True, is_anatomical=True))
        self.segments["Pelvis"].add_marker(Marker("RPSIS", is_technical=True, is_anatomical=True))
        self.segments["Pelvis"].add_marker(Marker("LASIS", is_technical=True, is_anatomical=True))
        self.segments["Pelvis"].add_marker(Marker("RASIS", is_technical=True, is_anatomical=True))

        self.segments["Thorax"] = Segment(
            name="Thorax",
            parent_name="Pelvis",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=self._thorax_joint_center,
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: (m["T10"] + m["STR"]) / 2,
                    end=lambda m, bio: (m["C7"] + m["SUP"]) / 2,
                ),
                second_axis=Axis(
                    Axis.Name.X,
                    start=lambda m, bio: (m["T10"] + m["C7"]) / 2,
                    end=lambda m, bio: (m["STR"] + m["SUP"]) / 2,
                ),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(("T10", "C7", "SUP", "STR", "T10")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.355 * self.body_mass,
                center_of_mass=self._thorax_center_of_mass,
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.355 * self.body_mass,
                    coef=(0.4, 0.5, 0.25),
                    start=m["C7"],
                    end=self._lumbar_5(m, bio),
                ),
            ),
        )
        self.segments["Thorax"].add_marker(Marker("T10", is_technical=True, is_anatomical=True))
        self.segments["Thorax"].add_marker(Marker("C7", is_technical=True, is_anatomical=True))
        self.segments["Thorax"].add_marker(Marker("STR", is_technical=True, is_anatomical=True))
        self.segments["Thorax"].add_marker(Marker("SUP", is_technical=True, is_anatomical=True))
        #self.segments["Thorax"].add_marker(Marker("RBAK", is_technical=True, is_anatomical=True))

        self.segments["Head"] = Segment(
            name="Head",
            parent_name="Thorax",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=self._head_joint_center,
                first_axis=Axis(
                    Axis.Name.X,
                    start="OCC",
                    end="SEL",
                ),
                second_axis=Axis(Axis.Name.Y, start="RTEMP", end="LTEMP"),
                axis_to_keep=Axis.Name.X,
            ),
            mesh=Mesh(("OCC", "RTEMP", "SEL", "LTEMP", "OCC", "HV")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.082 * self.body_mass,
                center_of_mass=self._head_center_of_mass,
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.082 * self.body_mass,
                    coef=(0.3, 0.3, 0.3),
                    start=m["HV"],
                    end=m["C7"],
                ),
            ),
        )
        self.segments["Head"].add_marker(Marker("OCC", is_technical=True, is_anatomical=True))
        self.segments["Head"].add_marker(Marker("LTEMP", is_technical=True, is_anatomical=True))
        self.segments["Head"].add_marker(Marker("RTEMP", is_technical=True, is_anatomical=True))
        self.segments["Head"].add_marker(Marker("SEL", is_technical=True, is_anatomical=True))
        self.segments["Head"].add_marker(Marker("HV", is_technical=True, is_anatomical=True))

        self.segments["RHumerus"] = Segment(
            name="RHumerus",
            parent_name="Thorax",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._humerus_joint_center(m, bio, "R"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: self._elbow_joint_center(m, bio, "R"),
                    end=lambda m, bio: self._humerus_joint_center(m, bio, "R"),
                ),
                second_axis=Axis(
                    Axis.Name.X,
                    start=lambda m, bio: self._elbow_joint_center(m, bio, "R"),
                    end=lambda m, bio: self._wrist_joint_center(m, bio, "R"),
                ),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._humerus_joint_center(m, bio, "R"),
                    lambda m, bio: self._elbow_joint_center(m, bio, "R"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.0271 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.5754, start=self._humerus_joint_center(m, bio, "R"), end=self._elbow_joint_center(m, bio, "R")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.0271 * self.body_mass,
                    coef=(0.33, 0.33, 0.20),
                    start=self._humerus_joint_center(m, bio, "R"),
                    end=self._elbow_joint_center(m, bio, "R"),
                ),
            ),
        )
        self.segments["RHumerus"].add_marker(Marker("RA", is_technical=True, is_anatomical=True))
        self.segments["RHumerus"].add_marker(Marker("RLHE", is_technical=True, is_anatomical=True))
        self.segments["RHumerus"].add_marker(Marker("RMHE", is_technical=True, is_anatomical=True))

        self.segments["RRadius"] = Segment(
            name="RRadius",
            parent_name="RHumerus",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._elbow_joint_center(m, bio, "R"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: self._wrist_joint_center(m, bio, "R"),
                    end=lambda m, bio: self._elbow_joint_center(m, bio, "R"),
                ),
                second_axis=Axis(
                    Axis.Name.Y,
                    start=lambda m, bio: bio.segments["RHumerus"].segment_coordinate_system.scs[:, 3, :],
                    end=lambda m, bio: bio.segments["RHumerus"].segment_coordinate_system.scs[:, 1, :],
                ),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._elbow_joint_center(m, bio, "R"),
                    lambda m, bio: self._wrist_joint_center(m, bio, "R"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.0162 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.57, start=self._elbow_joint_center(m, bio, "R"), end=self._wrist_joint_center(m, bio, "R")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.0162 * self.body_mass,
                    coef=(0.28,0.28,0.16),
                    start=self._elbow_joint_center(m, bio, "R"),
                    end=self._wrist_joint_center(m, bio, "R"),
                ),
            ),
        )
        self.segments["RRadius"].add_marker(Marker("RUS", is_technical=True, is_anatomical=True))
        self.segments["RRadius"].add_marker(Marker("RRS", is_technical=True, is_anatomical=True))

        self.segments["RHand"] = Segment(
            name="RHand",
            parent_name="RRadius",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._wrist_joint_center(m, bio, "R"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: self._hand_center(m, bio, "R"),
                    end=lambda m, bio: self._wrist_joint_center(m, bio, "R"),
                ),
                second_axis=Axis(Axis.Name.Y, start="RUS", end="RRS"),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh((lambda m, bio: self._wrist_joint_center(m, bio, "R"), "RFT3")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.006 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.7474,
                    start=self._wrist_joint_center(m, bio, "R"),
                    end=point_on_vector(1 / 0.75, start=self._wrist_joint_center(m, bio, "R"), end=m[f"RFT3"]),
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.006 * self.body_mass,
                    coef=(0.25,0.25,0.16),
                    start=self._wrist_joint_center(m, bio, "R"),
                    end=m[f"RFT3"],
                ),
            ),
        )
        self.segments["RHand"].add_marker(Marker("RFT3", is_technical=True, is_anatomical=True))
        self.segments["RHand"].add_marker(Marker("RHMH2", is_technical=True, is_anatomical=True))
        self.segments["RHand"].add_marker(Marker("RHMH5", is_technical=True, is_anatomical=True))

        self.segments["LHumerus"] = Segment(
            name="LHumerus",
            parent_name="Thorax",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._humerus_joint_center(m, bio, "L"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: self._elbow_joint_center(m, bio, "L"),
                    end=lambda m, bio: self._humerus_joint_center(m, bio, "L"),
                ),
                second_axis=Axis(
                    Axis.Name.X,
                    start=lambda m, bio: self._elbow_joint_center(m, bio, "L"),
                    end=lambda m, bio: self._wrist_joint_center(m, bio, "L"),
                ),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._humerus_joint_center(m, bio, "L"),
                    lambda m, bio: self._elbow_joint_center(m, bio, "L"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.0271 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.564, start=self._humerus_joint_center(m, bio, "L"), end=self._elbow_joint_center(m, bio, "L")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.0271 * self.body_mass,
                    coef=(0.33,0.33,0.2),
                    start=self._humerus_joint_center(m, bio, "L"),
                    end=self._elbow_joint_center(m, bio, "L"),
                ),
            ),
        )
        self.segments["LHumerus"].add_marker(Marker("LA", is_technical=True, is_anatomical=True))
        self.segments["LHumerus"].add_marker(Marker("LLHE", is_technical=True, is_anatomical=True))
        # TODO: Add ELBM to define the axis
        self.segments["LHumerus"].add_marker(Marker("LMHE", is_technical=True, is_anatomical=True))

        self.segments["LRadius"] = Segment(
            name="LRadius",
            parent_name="LHumerus",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._elbow_joint_center(m, bio, "L"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: self._wrist_joint_center(m, bio, "L"),
                    end=lambda m, bio: self._elbow_joint_center(m, bio, "L"),
                ),
                second_axis=Axis(
                    Axis.Name.Y,
                    start=lambda m, bio: bio.segments["LHumerus"].segment_coordinate_system.scs[:, 3, :],
                    end=lambda m, bio: bio.segments["LHumerus"].segment_coordinate_system.scs[:, 1, :],
                ),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._elbow_joint_center(m, bio, "L"),
                    lambda m, bio: self._wrist_joint_center(m, bio, "L"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.0162 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.4559, start=self._elbow_joint_center(m, bio, "L"), end=self._wrist_joint_center(m, bio, "L")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.0162 * self.body_mass,
                    coef=(0.28,0.28,0.16),
                    start=self._elbow_joint_center(m, bio, "L"),
                    end=self._wrist_joint_center(m, bio, "L"),
                ),
            ),
        )
        self.segments["LRadius"].add_marker(Marker("LUS", is_technical=True, is_anatomical=True))
        self.segments["LRadius"].add_marker(Marker("LRS", is_technical=True, is_anatomical=True))

        self.segments["LHand"] = Segment(
            name="LHand",
            parent_name="LRadius",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._wrist_joint_center(m, bio, "L"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: self._hand_center(m, bio, "L"),
                    end=lambda m, bio: self._wrist_joint_center(m, bio, "L"),
                ),
                second_axis=Axis(Axis.Name.Y, start="LUS", end="LRS"),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh((lambda m, bio: self._wrist_joint_center(m, bio, "L"), "LFT3")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.006 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.6205, start=self._wrist_joint_center(m, bio, "L"), end=m[f"LFT3"]
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.006 * self.body_mass,
                    coef=(0.25,0.25,0.16),
                    start=self._wrist_joint_center(m, bio, "L"),
                    end=m[f"LFT3"],
                ),
            ),
        )
        self.segments["LHand"].add_marker(Marker("LFT3", is_technical=True, is_anatomical=True))
        self.segments["LHand"].add_marker(Marker("LHMH2", is_technical=True, is_anatomical=True))
        self.segments["LHand"].add_marker(Marker("LHMH5", is_technical=True, is_anatomical=True))

        self.segments["RFemur"] = Segment(
            name="RFemur",
            parent_name="Pelvis",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._hip_joint_center(m, bio, "R"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: self._knee_joint_center(m, bio, "R"),
                    end=lambda m, bio: self._hip_joint_center(m, bio, "R"),
                ),
                second_axis=self._knee_axis("R"),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._hip_joint_center(m, bio, "R"),
                    lambda m, bio: self._knee_joint_center(m, bio, "R"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio:  0.142 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.3612, start=self._hip_joint_center(m, bio, "R"), end=self._knee_joint_center(m, bio, "R")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.142 * self.body_mass,
                    coef=(0.32, 0.32, 0.16),
                    start=self._hip_joint_center(m, bio, "R"),
                    end=self._knee_joint_center(m, bio, "R"),
                ),
            ),
        )
        self.segments["RFemur"].add_marker(Marker("RGT", is_technical=True, is_anatomical=True))
        self.segments["RFemur"].add_marker(Marker("RLFE", is_technical=True, is_anatomical=True))
        self.segments["RFemur"].add_marker(Marker("RMFE", is_technical=True, is_anatomical=True))

        self.segments["RTibia"] = Segment(
            name="RTibia",
            parent_name="RFemur",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._knee_joint_center(m, bio, "R"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: self._ankle_joint_center(m, bio, "R"),
                    end=lambda m, bio: self._knee_joint_center(m, bio, "R"),
                ),
                second_axis=self._knee_axis("R"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._knee_joint_center(m, bio, "R"),
                    lambda m, bio: self._ankle_joint_center(m, bio, "R"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.0433 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.4416, start=self._knee_joint_center(m, bio, "R"), end=self._ankle_joint_center(m, bio, "R")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.0433 * self.body_mass,
                    coef=(0.3, 0.3, 0.2),
                    start=self._knee_joint_center(m, bio, "R"),
                    end=self._ankle_joint_center(m, bio, "R"),
                ),
            ),
        )
        self.segments["RTibia"].add_marker(Marker("RLM", is_technical=True, is_anatomical=True))
        self.segments["RTibia"].add_marker(Marker("RSPH", is_technical=True, is_anatomical=True))
        self.segments["RTibia"].add_marker(Marker("RATT", is_technical=True, is_anatomical=True))


        self.segments["RFoot"] = Segment(
            name="RFoot",
            parent_name="RTibia",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._ankle_joint_center(m, bio, "R"),
                first_axis=Axis(Axis.Name.Y, start="RLM", end="RSPH"),
                second_axis=Axis(Axis.Name.Z, start="RTT2", end="RCAL"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("RTT2", "RMFH5", "RLM", "RCAL", "RSPH", "RMFH1", "RTT2")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.0133 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.5, start=m[f"RCAL"], end=m[f"RTT2"]
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.0133 * self.body_mass,
                    coef=(0.25, 0.25, 0.15),
                    start=self._ankle_joint_center(m, bio, "R"),
                    end=m[f"RTT2"],
                ),
            ),
        )
        self.segments["RFoot"].add_marker(Marker("RTT2", is_technical=True, is_anatomical=True))
        self.segments["RFoot"].add_marker(Marker("RMFH5", is_technical=True, is_anatomical=True))
        self.segments["RFoot"].add_marker(Marker("RCAL", is_technical=True, is_anatomical=True))
        #self.segments["RFoot"].add_marker(Marker("RLM", is_technical=True, is_anatomical=True))
        #self.segments["RFoot"].add_marker(Marker("RSPH", is_technical=True, is_anatomical=True))
        self.segments["RFoot"].add_marker(Marker("RMFH1", is_technical=True, is_anatomical=True))

        self.segments["LFemur"] = Segment(
            name="LFemur",
            parent_name="Pelvis",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._hip_joint_center(m, bio, "L"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: self._knee_joint_center(m, bio, "L"),
                    end=lambda m, bio: self._hip_joint_center(m, bio, "L"),
                ),
                second_axis=self._knee_axis("L"),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._hip_joint_center(m, bio, "L"),
                    lambda m, bio: self._knee_joint_center(m, bio, "L"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.142 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.3612, start=self._hip_joint_center(m, bio, "L"), end=self._knee_joint_center(m, bio, "L")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.142 * self.body_mass,
                    coef=(0.32, 0.32, 0.16),
                    start=self._hip_joint_center(m, bio, "L"),
                    end=self._knee_joint_center(m, bio, "L"),
                ),
            ),
        )
        self.segments["LFemur"].add_marker(Marker("LGT", is_technical=True, is_anatomical=True))
        self.segments["LFemur"].add_marker(Marker("LLFE", is_technical=True, is_anatomical=True))
        self.segments["LFemur"].add_marker(Marker("LMFE", is_technical=True, is_anatomical=True))

        self.segments["LTibia"] = Segment(
            name="LTibia",
            parent_name="LFemur",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._knee_joint_center(m, bio, "L"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: self._ankle_joint_center(m, bio, "L"),
                    end=lambda m, bio: self._knee_joint_center(m, bio, "L"),
                ),
                second_axis=self._knee_axis("L"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._knee_joint_center(m, bio, "L"),
                    lambda m, bio: self._ankle_joint_center(m, bio, "L"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.0433 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.5, start=self._knee_joint_center(m, bio, "L"), end=self._ankle_joint_center(m, bio, "L")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.0433 * self.body_mass,
                    coef=(0.3, 0.3, 0.2),
                    start=self._knee_joint_center(m, bio, "L"),
                    end=self._ankle_joint_center(m, bio, "L"),
                ),
            ),
        )
        self.segments["LTibia"].add_marker(Marker("LLM", is_technical=True, is_anatomical=True))
        self.segments["LTibia"].add_marker(Marker("LSPH", is_technical=True, is_anatomical=True))
        self.segments["LTibia"].add_marker(Marker("LATT", is_technical=True, is_anatomical=True))

        self.segments["LFoot"] = Segment(
            name="LFoot",
            parent_name="LTibia",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._ankle_joint_center(m, bio, "L"),
                first_axis=Axis(Axis.Name.Y, start="LSPH", end="LLM"),
                second_axis=Axis(Axis.Name.Z, start="LTT2", end="LCAL"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("LTT2", "LMFH5", "LLM", "LCAL", "LSPH", "LMFH1", "LTT2")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: 0.0133 * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    0.4014, start=m[f"LCAL"], end=m[f"LTT2"]
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=0.0133 * self.body_mass,
                    coef=(0.25, 0.25, 0.15),
                    start=self._ankle_joint_center(m, bio, "L"),
                    end=m[f"LTT2"],
                ),
            ),
        )
        self.segments["LFoot"].add_marker(Marker("LTT2", is_technical=True, is_anatomical=True))
        self.segments["LFoot"].add_marker(Marker("LMFH5", is_technical=True, is_anatomical=True))
        self.segments["LFoot"].add_marker(Marker("LCAL", is_technical=True, is_anatomical=True))
        #self.segments["LFoot"].add_marker(Marker("LLM", is_technical=True, is_anatomical=True))
        #self.segments["LFoot"].add_marker(Marker("LSPH", is_technical=True, is_anatomical=True))
        self.segments["LFoot"].add_marker(Marker("LMFH1", is_technical=True, is_anatomical=True))

    def _lumbar_5(self, m, bio):
        right_hip = self._hip_joint_center(m, bio, "R")
        left_hip = self._hip_joint_center(m, bio, "L")
        return np.nanmean((left_hip, right_hip), axis=0) + np.array((0.0, 0.0, 0.828, 0))[:, np.newaxis] * np.repeat(
            np.linalg.norm(left_hip - right_hip, axis=0)[np.newaxis, :], 4, axis=0
        )

    def _pelvis_joint_center(self, m: dict, bio: BiomechanicalModelReal):
        return (m["LPSIS"] + m["RPSIS"] + m["LASIS"] + m["RASIS"]) / 4

    def _pelvis_center_of_mass(self, m: dict, bio: BiomechanicalModelReal) -> np.ndarray:
        """
        This computes the center of mass of the thorax

        Parameters
        ----------
        m
            The marker positions in the static
        bio
            The BiomechanicalModelReal as it is constructed so far
        """
        right_hip = self._hip_joint_center(m, bio, "R")
        left_hip = self._hip_joint_center(m, bio, "L")
        p = self._pelvis_joint_center(m, bio)  # Make sur the center of mass is symmetric
        p[2, :] += 0.925 * (self._lumbar_5(m, bio) - np.nanmean((left_hip, right_hip), axis=0))[2, :]
        return p

    def _thorax_joint_center(self, m: dict, bio: BiomechanicalModelReal):
        return m["SUP"]

    def _thorax_center_of_mass(self, m: dict, bio: BiomechanicalModelReal) -> np.ndarray:
        """
        This computes the center of mass of the thorax

        Parameters
        ----------
        m
            The marker positions in the static
        bio
            The BiomechanicalModelReal as it is constructed so far
        """
        com = point_on_vector(0.63, start=m["C7"], end=self._lumbar_5(m, bio))
        com[0, :] = self._thorax_joint_center(m, bio)[0, :]  # Make sur the center of mass is symmetric
        return com

    def _head_joint_center(self, m: dict, bio: BiomechanicalModelReal):
        return (m["LTEMP"] + m["RTEMP"]) / 2

    def _head_center_of_mass(self, m: dict, bio: BiomechanicalModelReal):
        return point_on_vector(
            0.52,
            start=m["SEL"],
            end=m["OCC"],
        )

    def _humerus_joint_center(self, m: dict, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        This is the implementation of the 'Shoulder joint center, p.69'.

        Parameters
        ----------
        m
            The marker positions in the static
        bio
            The BiomechanicalModelReal as it is constructed so far
        side
            If the markers are from the right ("R") or left ("L") side

        Returns
        -------
        The position of the origin of the humerus
        """

        thorax_origin = bio.segments["Thorax"].segment_coordinate_system.scs[:, 3, :]
        thorax_x_axis = bio.segments["Thorax"].segment_coordinate_system.scs[:, 0, :]
        thorax_to_sho_axis = m[f"{side}A"] - thorax_origin
        shoulder_wand = np.cross(thorax_to_sho_axis[:3, :], thorax_x_axis[:3, :], axis=0)
        shoulder_offset = (
            self.shoulder_offset
            if self.shoulder_offset is not None
            else 0.17 * (m[f"{side}A"] - m[f"{side}LHE"])[2, :]
        )

        return chord_function(shoulder_offset, thorax_origin, m[f"{side}A"], shoulder_wand)

    def _elbow_joint_center(self, m: dict, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        Compute the joint center of

        Parameters
        ----------
        m
            The marker positions in the static
        bio
            The BiomechanicalModelReal as it is constructed so far
        side
            If the markers are from the right ("R") or left ("L") side

        Returns
        -------
        The position of the origin of the elbow
        """

        shoulder_origin = self._humerus_joint_center(m, bio, side)
        elbow_marker = (m[f"{side}LHE"] + m[f"{side}MHE"]) / 2
        wrist_marker = (m[f"{side}RS"] + m[f"{side}US"]) / 2

        elbow_width = (
            self.elbow_width
            if self.elbow_width is not None
            else np.linalg.norm(m[f"{side}RS"][:3, :] - m[f"{side}US"][:3, :], axis=0) * 1.15
        )
        elbow_offset = elbow_width / 2

        return chord_function(elbow_offset, shoulder_origin, elbow_marker, wrist_marker)

    def _wrist_joint_center(self, m, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        Compute the segment coordinate system of the wrist. If wrist_width is not provided 2cm is assumed

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side

        Returns
        -------
        The SCS of the wrist
        """

        elbow_center = self._elbow_joint_center(m, bio, side)
        wrist_bar_center = project_point_on_line(m[f"{side}RS"], m[f"{side}US"], elbow_center)
        offset_axis = np.cross(
            m[f"{side}RS"][:3, :] - m[f"{side}US"][:3, :], elbow_center[:3, :] - wrist_bar_center, axis=0
        )
        offset_axis /= np.linalg.norm(offset_axis, axis=0)

        offset = (offset_axis * (self.wrist_width / 2)) if self.wrist_width is not None else 0.02 / 2
        return np.concatenate((wrist_bar_center - offset, np.ones((1, wrist_bar_center.shape[1])))) #wrist_bar_center + offset

    def _hand_center(self, m, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        Compute the origin of the hand. If hand_thickness if not provided, it is assumed to be 1cm

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """

        elbow_center = self._elbow_joint_center(m, bio, side)
        wrist_joint_center = self._wrist_joint_center(m, bio, side)
        fin_marker = m[f"{side}FT3"]
        hand_offset = np.repeat(self.hand_thickness / 2 if self.hand_thickness else 0.01 / 2, fin_marker.shape[1])
        wrist_bar_center = project_point_on_line(m[f"{side}RS"], m[f"{side}US"], elbow_center)

        return chord_function(hand_offset, wrist_joint_center, fin_marker, wrist_bar_center)

    def _legs_length(self, m, bio: BiomechanicalModelReal):
        # TODO: Verify 95% makes sense
        return {
            "R": self.leg_length["R"] if self.leg_length else np.nanmean(np.linalg.norm(m["RGT"][:3, :]-m["RLM"][:3, :], axis=0)),
            "L": self.leg_length["L"] if self.leg_length else np.nanmean(np.linalg.norm(m["LGT"][:3, :]-m["LLM"][:3, :], axis=0)),
        }

    def _hip_joint_center(self, m, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        Compute the hip joint center. The LegLength is not provided, the height of the TROC is used (therefore assuming
        the subject is standing upright during the static trial)

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """
        """
               inter_asis = np.nanmean(np.linalg.norm(m["LASIS"][:3, :] - m["RASIS"][:3, :], axis=0))
               legs_length = self._legs_length(m, bio)
               mean_legs_length = np.nanmean((legs_length["R"], legs_length["L"]))
               asis_troc_dist = 0.1288 * legs_length[side] - 0.04856

               c = mean_legs_length * 0.115 - 0.0153
               aa = inter_asis / 2
               theta = 0.5
               beta = 0.314
               x = c * np.cos(theta) * np.sin(beta) - asis_troc_dist * np.cos(beta)
               y = -(c * np.sin(theta) - aa)
               z = -c * np.cos(theta) * np.cos(beta) - asis_troc_dist * np.sin(beta)
               return m[f"{side}ASIS"] + np.array((x, y, z, 0))[:, np.newaxis]

        """
        inter_asis = np.nanmean(np.linalg.norm(m["LASIS"][:3, :] - m["RASIS"][:3, :], axis=0))
        legs_length = self._legs_length(m, bio)
        PJC = self._pelvis_joint_center(m, bio)

        mean_legs_length = np.nanmean((legs_length["R"], legs_length["L"]))
        asis_troc_dist = 0.1288 * legs_length[side] - 0.04856
        #asis_troc_dist = np.nanmean(np.linalg.norm(m["RGT"][:3, :] - m["RASIS"][:3, :], axis=0))
        x = 0.011-0.063 * mean_legs_length
        y = 8/1000 + 0.086 * mean_legs_length
        z = -9/1000 - 0.078 * mean_legs_length
        Axe = m[f"{side}ASIS"]-PJC
        dir = np.mean(Axe[1,:])/np.abs(np.mean(Axe[1,:]))
        x = PJC[0,:] - x
        y = PJC[1,:] + y*dir
        z = PJC[2,:] + z
        return np.array((x, y, z, m[f"{side}ASIS"][3,:])) #m[f"{side}ASIS"] + (np.array((x, y, z, 0))[:, np.newaxis]/2)


    def _knee_axis(self, side) -> Axis:
        """
        Define the knee axis

        Parameters
        ----------
        side
            If the markers are from the right ("R") or left ("L") side
        """
        if side == "R":
            return Axis(Axis.Name.Y, start=f"{side}LFE", end=f"{side}MFE")
        elif side == "L":
            return Axis(Axis.Name.Y, start=f"{side}MFE", end=f"{side}LFE")
        else:
            raise ValueError("side should be 'R' or 'L'")

    def _knee_joint_center(self, m, bio: BiomechanicalModelReal, side) -> np.ndarray:
        """
        Compute the knee joint center. This is a simplified version since the KNM exists

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """
        return (m[f"{side}MFE"] + m[f"{side}LFE"]) / 2

    def _ankle_joint_center(self, m, bio: BiomechanicalModelReal, side) -> np.ndarray:
        """
        Compute the ankle joint center. This is a simplified version sie ANKM exists

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """

        return (m[f"{side}SPH"] + m[f"{side}LM"]) / 2

    @property
    def dof_index(self) -> dict[str, tuple[int, ...]]:
        """
        Returns a dictionary with all the dof to export to the C3D and their corresponding XYZ values in the generalized
        coordinate vector
        """

        # TODO: Some of these values as just copy of their relative
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
