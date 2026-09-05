---
notion:
  role: bonus
  scope: demo
  status: mapped
  page_id: "3d2d9fdd-1a1a-815e-ab78-c339d96282f1"
  url: "https://app.notion.com/p/3d2d9fdd1a1a815eab78c339d96282f1"
---

# Optional Geography Demo

`05_geo_bonus.ipynb` is a demo-only, non-graded extension. It is not assumed prior
knowledge and is not referenced by assignment requirements or the grader.

[Open the optional geo notebook in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/11/demo/05_geo_bonus.ipynb)

## Course concepts

The course concepts remain familiar: read a small results table, validate a join,
choose an honest summary measure, and label a static figure clearly.

## Supplied geospatial machinery

The notebook supplies the unfamiliar mechanics: installing separately pinned geo
packages, downloading official TLC taxi-zone polygons, reading a shapefile, joining
on zone ID, and drawing polygon geometry. Students are not expected to reproduce
that machinery in the assignment.

Polygon-only rendering is the successful default. An OpenStreetMap basemap could
be added as an optional enhancement, but it would introduce tile-network and
projection dependencies and is not required here.

Run Demo 4 first so `output/04_zone_error_summary.csv` exists, then run
`05_geo_bonus.ipynb`. The required four demos do not import geo packages and do not
depend on this notebook.
