from enum import Enum


class ModelImports(Enum):
    # Line
    KRAKENLINE = ("src.tasks.line.kraken_line", "KrakenLineTask")
    YOLOLINE = ("src.tasks.line.yolo_line", "YoloLineTask")

    # Layout
    YOLOLAYOUT = ("src.tasks.layout.yolo_layout", "YoloLayoutTask")
    
    # HTR
    KRAKENHTR = ("src.tasks.htr.kraken_htr", "KrakenHTRTask")
    VLMHTR = ("src.tasks.htr.vlm_htr", "VLMHTRTask")
    TROCRHTR = ("src.tasks.htr.trocr_htr", "TrOCRHTRTask")