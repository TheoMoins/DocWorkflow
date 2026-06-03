# ALTO/XML Data Format

This document describes the ALTO/XML input format expected by `docworkflow`.
It is intended both for users who want to prepare their data and for LLMs tasked
with converting an existing document into this format.

---

## 1. Overview

The pipeline accepts files in **ALTO XML v4** format (Analyzed Layout and Text Object),
a standard developed by the Library of Congress. Each XML file describes the layout of
a handwritten or printed page image and contains (depending on the mode) the transcription
associated with each line.

**Required namespace:**
```
http://www.loc.gov/standards/alto/ns-v4#
```

**Minimal XML header:**
```xml
<?xml version='1.0' encoding='UTF-8'?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4#
                          http://www.loc.gov/standards/alto/v4/alto-4-2.xsd">
```

The pipeline operates in two distinct modes depending on the granularity of the input:

| Mode | Description |
|------|-------------|
| **Line level** | Lines are pre-segmented in the ALTO file; each line is individually cropped from the image for transcription |
| **Page level** | Only the full page image is used; segmentation is produced as output, not provided as input |

---

## 2. General Structure of an ALTO File

```xml
<alto>
  <Description>
    <MeasurementUnit>pixel</MeasurementUnit>
    <sourceImageInformation>
      <fileName>image_name.jpg</fileName>
    </sourceImageInformation>
  </Description>

  <Tags>
    <OtherTag ID="BT1" LABEL="MainZone"       DESCRIPTION="block type MainZone"/>
    <OtherTag ID="BT2" LABEL="MarginTextZone" DESCRIPTION="block type MarginTextZone"/>
    <OtherTag ID="LT1" LABEL="DefaultLine"    DESCRIPTION="line type DefaultLine"/>
    <OtherTag ID="LT2" LABEL="HeadingLine"    DESCRIPTION="line type HeadingLine"/>
  </Tags>

  <Layout>
    <Page ID="page1" PHYSICAL_IMG_NR="1" HEIGHT="2836" WIDTH="2000">
      <PrintSpace HEIGHT="2836" WIDTH="2000" VPOS="0" HPOS="0">
        <!-- TextBlock(s) here -->
      </PrintSpace>
    </Page>
  </Layout>
</alto>
```

### `Description` Section

| Element | Role |
|---------|------|
| `MeasurementUnit` | Must be `pixel` — all coordinates are in pixels |
| `sourceImageInformation/fileName` | Name of the associated image file (path relative to the XML folder) |

### `Tags` Section

Each `OtherTag` defines a reusable label via its `ID` attribute.
`TextBlock` and `TextLine` elements reference these labels via the `TAGREFS` attribute.

Two tag families:
- **Block types** (prefix `BT`): text zone type
- **Line types** (prefix `LT`): line type

#### Zone types ignored by the pipeline

`TextBlock` elements whose `LABEL` is one of the following are **ignored** during processing
(no transcription, no evaluation):

```
DigitizationArtefactZone
MarginTextZone
NumberingZone
DropCapitalZone
```

### `Layout` Section

| Element | Key attributes |
|---------|---------------|
| `Page` | `HEIGHT`, `WIDTH` (dimensions in pixels), `ID`, `PHYSICAL_IMG_NR` |
| `PrintSpace` | `HEIGHT`, `WIDTH`, `VPOS`, `HPOS` (typically 0, 0) |

---

## 3. Line Level Mode

### Principle

The ALTO file must contain the full page segmentation: text zones (`TextBlock`) and lines
(`TextLine`) with precise geometric coordinates. The pipeline crops each line from the source
image and transcribes it independently.

### Expected Structure

```xml
<TextBlock ID="block_0"
           HPOS="504" VPOS="383" WIDTH="999" HEIGHT="1899"
           TAGREFS="BT1">
  <Shape>
    <Polygon POINTS="504,383 1503,383 1503,2282 504,2282"/>
  </Shape>

  <TextLine ID="_58afa90e-8029-4333-a730-6968ed513f84"
            TAGREFS="LT1"
            HPOS="112" VPOS="75" WIDTH="513" HEIGHT="58"
            BASELINE="113 116 625 110">
    <Shape>
      <Polygon POINTS="112,75 625,75 625,133 112,133"/>
    </Shape>
    <String CONTENT="Cal nome pohansky fizilmi" WC="1.0"/>
  </TextLine>

  <TextLine ID="_a4f2c1b0-..." TAGREFS="LT1"
            HPOS="115" VPOS="145" WIDTH="498" HEIGHT="55"
            BASELINE="115 170 613 168">
    <Shape>
      <Polygon POINTS="115,145 613,145 613,200 115,200"/>
    </Shape>
    <String CONTENT="Autem vero dicebat hec" WC="1.0"/>
  </TextLine>
</TextBlock>
```

### `TextBlock` Attributes

| Attribute | Type | Role |
|-----------|------|------|
| `ID` | string | Unique block identifier |
| `HPOS` | integer | Horizontal position in pixels (top-left corner) |
| `VPOS` | integer | Vertical position in pixels (top-left corner) |
| `WIDTH` | integer | Width in pixels |
| `HEIGHT` | integer | Height in pixels |
| `TAGREFS` | string | Reference to the block type tag ID (e.g. `BT1`) |

A `TextBlock` may optionally contain a `Shape/Polygon` (same format as for lines).

### `TextLine` Attributes

| Attribute | Type | Required | Role |
|-----------|------|----------|------|
| `ID` | string | **Yes** | Unique, stable line identifier |
| `HPOS` | integer | **Yes** | Horizontal position of the bounding box |
| `VPOS` | integer | **Yes** | Vertical position of the bounding box |
| `WIDTH` | integer | **Yes** | Bounding box width |
| `HEIGHT` | integer | **Yes** | Bounding box height |
| `BASELINE` | string | **Yes*** | Baseline coordinates (see format below) |
| `TAGREFS` | string | No | Reference to the line type |

\* `BASELINE` is mandatory in line level mode (at least 2 points required).

#### `BASELINE` Attribute Format

Alternating `x y x y ...` coordinates (space-separated, **no commas**):

```
BASELINE="x1 y1 x2 y2 [x3 y3 ...]"
```

Example: `BASELINE="113 116 625 110"` = baseline from `(113, 116)` to `(625, 110)`.

A minimum of **2 points** (4 values) is required.

### Contour Polygon (`Shape/Polygon`)

Optional but **strongly recommended** child element of `TextLine`.
Defines the precise boundary of the text zone for cropping.

```xml
<Shape>
  <Polygon POINTS="x1,y1 x2,y2 x3,y3 x4,y4"/>
</Shape>
```

**Two accepted formats for `POINTS`:**

| Format | Example |
|--------|---------|
| Comma-pair (preferred) | `"112,75 625,75 625,133 112,133"` |
| Alternating coordinates | `"112 75 625 75 625 133 112 133"` |

The bounding box is automatically computed as the envelope of the polygon.

### `String` Element (transcription)

```xml
<String CONTENT="transcribed text" WC="1.0"/>
```

| Attribute | Role |
|-----------|------|
| `CONTENT` | Transcribed text for this line |
| `WC` | Confidence score (float between 0.0 and 1.0; 1.0 = ground truth) |

### Geometry Fallback Chain

If a `TextLine` is incomplete, the pipeline applies the following fallback chain:

```
Shape/Polygon  →  BASELINE  →  HPOS + VPOS + WIDTH + HEIGHT
```

As a last resort (HPOS/VPOS/WIDTH/HEIGHT only), a synthetic horizontal baseline is
generated at the vertical midpoint of the bounding box.

---

## 4. Page Level Mode

### Principle

In page level mode, the model transcribes the entire page at once. The input ALTO file
**does not need to contain `TextLine` elements**: only the text zones (`TextBlock`) are used
to guide segmentation, and the transcription is then split into lines automatically.

If no ALTO file is provided, a minimal ALTO covering the entire image is generated automatically.

### Minimal Acceptable Format

```xml
<?xml version='1.0' encoding='UTF-8'?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4#
                          http://www.loc.gov/standards/alto/v4/alto-4-2.xsd">
  <Description>
    <MeasurementUnit>pixel</MeasurementUnit>
    <sourceImageInformation>
      <fileName>image-0001.jpg</fileName>
    </sourceImageInformation>
  </Description>
  <Tags>
    <OtherTag ID="BT1" LABEL="MainZone"    DESCRIPTION="block type MainZone"/>
    <OtherTag ID="LT1" LABEL="DefaultLine" DESCRIPTION="line type DefaultLine"/>
  </Tags>
  <Layout>
    <Page ID="page1" PHYSICAL_IMG_NR="1" HEIGHT="2836" WIDTH="2000">
      <PrintSpace HEIGHT="2836" WIDTH="2000" VPOS="0" HPOS="0">
        <TextBlock ID="block_0"
                   HPOS="504" VPOS="383" WIDTH="999" HEIGHT="1899"
                   TAGREFS="BT1">
          <Shape>
            <Polygon POINTS="504,383 1503,383 1503,2282 504,2282"/>
          </Shape>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
```

Notes:
- `TextBlock` elements define the zones that will be transcribed.
- `TextLine` elements are absent in input; they will be created in the output by the pipeline.
- A `TextBlock` without a `Shape/Polygon` is accepted if `HPOS`, `VPOS`, `WIDTH`, and `HEIGHT` are present.

### ALTO Output Format (page level)

After transcription, the pipeline produces an ALTO with `TextLine` elements split by newlines:

```xml
<TextBlock ID="block_0" HPOS="0" VPOS="0" WIDTH="2000" HEIGHT="2836" TAGREFS="BT1">
  <TextLine ID="line_0" TAGREFS="LT1"
            HPOS="5" VPOS="5" WIDTH="1990" HEIGHT="130"
            BASELINE="5 70 1995 70">
    <Shape>
      <Polygon POINTS="5,5 1995,5 1995,135 5,135"/>
    </Shape>
    <String CONTENT="first line of text" WC="0.95"/>
  </TextLine>
  <TextLine ID="line_1" TAGREFS="LT1"
            HPOS="5" VPOS="140" WIDTH="1990" HEIGHT="130"
            BASELINE="5 205 1995 205">
    <Shape>
      <Polygon POINTS="5,140 1995,140 1995,270 5,270"/>
    </Shape>
    <String CONTENT="second line of text" WC="0.92"/>
  </TextLine>
</TextBlock>
```

---

## 5. Comparison of the Two Modes

| Criterion | Line Level | Page Level |
|-----------|-----------|------------|
| **Input segmentation** | Required (TextLine with geometry) | Optional (TextBlock is sufficient) |
| **BASELINE** | Required (≥ 2 points) | Absent in input, generated in output |
| **Shape/Polygon** | Strongly recommended | Absent in input, generated in output |
| **String/CONTENT** | Present (ground truth or empty) | Absent in input |
| **Image used** | Cropped per line | Full page |
| **Output segmentation** | Unchanged (text updated) | TextLines generated automatically |

---

## 6. Data Preparation Checklist

### Line level mode

- [ ] ALTO v4 namespace present in `<alto>`
- [ ] `<fileName>` points to the image in the same folder
- [ ] `MeasurementUnit` = `pixel`
- [ ] Each `TextBlock` has a unique `ID` and a valid `TAGREFS`
- [ ] Zones to be ignored have a label in `{DigitizationArtefactZone, MarginTextZone, NumberingZone, DropCapitalZone}`
- [ ] Each `TextLine` has a unique and stable `ID`
- [ ] Each `TextLine` has a `BASELINE` with at least 2 points (format `x1 y1 x2 y2`)
- [ ] Each `TextLine` has a `Shape/Polygon` (or at minimum `HPOS`, `VPOS`, `WIDTH`, `HEIGHT`)
- [ ] `String` elements have a `CONTENT` attribute (may be empty for prediction)

### Page level mode

- [ ] ALTO v4 namespace present in `<alto>`
- [ ] `<fileName>` points to the image in the same folder
- [ ] `Page` has correct `HEIGHT` and `WIDTH` (in pixels)
- [ ] At least one non-ignored `TextBlock` is present

---

## 7. Full Example (line level)

```xml
<?xml version='1.0' encoding='UTF-8'?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4#
                          http://www.loc.gov/standards/alto/v4/alto-4-2.xsd">
  <Description>
    <MeasurementUnit>pixel</MeasurementUnit>
    <sourceImageInformation>
      <fileName>folio_042.jpg</fileName>
    </sourceImageInformation>
  </Description>

  <Tags>
    <OtherTag ID="BT1" LABEL="MainZone"       DESCRIPTION="block type MainZone"/>
    <OtherTag ID="BT2" LABEL="MarginTextZone" DESCRIPTION="block type MarginTextZone"/>
    <OtherTag ID="LT1" LABEL="DefaultLine"    DESCRIPTION="line type DefaultLine"/>
    <OtherTag ID="LT2" LABEL="HeadingLine"    DESCRIPTION="line type HeadingLine"/>
  </Tags>

  <Layout>
    <Page ID="page1" PHYSICAL_IMG_NR="1" HEIGHT="3200" WIDTH="2400">
      <PrintSpace HEIGHT="3200" WIDTH="2400" VPOS="0" HPOS="0">

        <TextBlock ID="block_main" HPOS="300" VPOS="250" WIDTH="1600" HEIGHT="2600"
                   TAGREFS="BT1">
          <Shape>
            <Polygon POINTS="300,250 1900,250 1900,2850 300,2850"/>
          </Shape>

          <TextLine ID="line_001" TAGREFS="LT2"
                    HPOS="310" VPOS="260" WIDTH="1580" HEIGHT="70"
                    BASELINE="310 320 1890 318">
            <Shape>
              <Polygon POINTS="310,260 1890,260 1890,330 310,330"/>
            </Shape>
            <String CONTENT="Incipit liber primus de naturis rerum" WC="1.0"/>
          </TextLine>

          <TextLine ID="line_002" TAGREFS="LT1"
                    HPOS="312" VPOS="345" WIDTH="1575" HEIGHT="65"
                    BASELINE="312 400 1887 398">
            <Shape>
              <Polygon POINTS="312,345 1887,345 1887,410 312,410"/>
            </Shape>
            <String CONTENT="Omnium rerum principia quedam sunt" WC="1.0"/>
          </TextLine>

          <TextLine ID="line_003" TAGREFS="LT1"
                    HPOS="314" VPOS="425" WIDTH="1570" HEIGHT="65"
                    BASELINE="314 480 1884 479">
            <Shape>
              <Polygon POINTS="314,425 1884,425 1884,490 314,490"/>
            </Shape>
            <String CONTENT="ex quibus omnia constant" WC="1.0"/>
          </TextLine>
        </TextBlock>

        <!-- Margin zone: ignored during HTR processing -->
        <TextBlock ID="block_margin" HPOS="2050" VPOS="800" WIDTH="300" HEIGHT="200"
                   TAGREFS="BT2">
          <Shape>
            <Polygon POINTS="2050,800 2350,800 2350,1000 2050,1000"/>
          </Shape>
          <TextLine ID="line_m01" TAGREFS="LT1"
                    HPOS="2055" VPOS="810" WIDTH="290" HEIGHT="55"
                    BASELINE="2055 855 2345 853">
            <Shape>
              <Polygon POINTS="2055,810 2345,810 2345,865 2055,865"/>
            </Shape>
            <String CONTENT="nota bene" WC="1.0"/>
          </TextLine>
        </TextBlock>

      </PrintSpace>
    </Page>
  </Layout>
</alto>
```

---

## 8. Notes on Converting from Other Formats

If you are converting from another transcription format (PAGE XML, hOCR, TEI, etc.),
here are the main correspondences:

| Concept | PAGE XML | hOCR | ALTO v4 |
|---------|----------|------|---------|
| Text zone | `TextRegion` | `ocr_carea` | `TextBlock` |
| Line | `TextLine` | `ocr_line` | `TextLine` |
| Line text | `TextEquiv/Unicode` | `ocr_word` CONTENT | `String CONTENT` |
| Contour polygon | `Coords points="..."` | `title="bbox ..."` | `Shape/Polygon POINTS="..."` |
| Baseline | `Baseline points="..."` | — | `BASELINE="..."` |
| Confidence | `TextEquiv conf="..."` | `x_wconf` | `String WC="..."` |

**Key pitfall when converting PAGE XML → ALTO:**
- In PAGE XML, `Baseline` uses the format `x1,y1 x2,y2` (comma within pairs, space between pairs).
- In ALTO, `BASELINE` uses the format `x1 y1 x2 y2` (spaces only, no commas).
- In PAGE XML, `Coords` use `x1,y1 x2,y2` (comma-pair format).
- In ALTO, `Polygon POINTS` accept both formats (`x1,y1 x2,y2` or `x1 y1 x2 y2`).
