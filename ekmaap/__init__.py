"""EkMaap: onion grading from photos (SIH26031, team StrataCadastre_KP).

Core package shared by the training scripts and the backend API:

    calibration   printed ArUco mat -> homography -> top-down image in millimetres
    segmentation  onion masks: classical baseline, trainable pixel forest, optional YOLO-seg
    measure       widest diameter in mm from a mask
    features      colour / shape / texture numbers per onion
    classifier    condition (sound / sprouted / damaged / rotten): heuristics or trained model
    rules         versioned grading rule set -> Grade A / URS / Reject
    pipeline      photo -> analysis (all of the above)
    coco, evaluate, synth, annotate   data formats, metrics, synthetic test images, drawing
"""

__version__ = "0.2.0"

CONDITIONS = ("sound", "sprouted", "damaged", "rotten")
