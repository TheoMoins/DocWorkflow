from enum import Enum


class ModelImports(Enum):
    # Line
    KRAKENLINE = ("src.tasks.line.kraken_line", "KrakenLineTask")
    YOLOLINE = ("src.tasks.line.yolo_line", "YoloLineTask")

    # Layout
    YOLOLAYOUT = ("src.tasks.layout.yolo_layout", "YoloLayoutTask")
    
    # HTR
    KRAKENHTR = ("src.tasks.htr.kraken_htr", "KrakenHTRTask")
    VLMPAGEHTR = ("src.tasks.htr.vlm_page_htr", "VLMPageHTRTask")
    VLMLINEHTR = ("src.tasks.htr.vlm_line_htr", "VLMLineHTRTask")
    VLMMULTILINEHTR = ("src.tasks.htr.vlm_multiline_htr", "VLMMultiLineHTRTask")
    VLMSILVERHTR = ("src.tasks.htr.vlm_line_htr_silver", "VLMLineHTRTaskSilver")
    TROCRHTR = ("src.tasks.htr.trocr_htr", "TrOCRHTRTask")