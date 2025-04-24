#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the df
df = pd.read_csv('anx3dpia.csv', index_col=0)
df = df[::-1]

# Create the heatmap
colormap = sns.color_palette("Reds")
grays = sns.color_palette("Grays")
plt.figure(figsize=(10, 8))
scaled_df = df.div(1-df.max(axis=1), axis=0)
print(scaled_df)

# df.reset_index()
# # df = df[::-1]
# for index, row in df[::-1].iterrows():
#     print(index, row.iloc[0])
#     mask = np.zeros_like(df, dtype=np.bool)
#     mask[:,index] = True
#     if index % 2 == 0:
#         colormap = colormap1
#     else:
#         colormap = colormap2
#     sns.heatmap(df, annot=True, fmt='g', linewidth=0.5, cmap=colormap, square=True, cbar=False, mask=mask, vmin=0, vmax=row.iloc[0], center=row.iloc[0]/2)
sns.heatmap(scaled_df, annot=df, fmt='g', linewidth=0.5, cmap=colormap, square=True, cbar=False)

mask = np.ones_like(df, dtype=np.bool)
mask[:,0] = False
sns.heatmap(df, annot=True, fmt='g', linewidth=0.5, cmap=grays, square=True, cbar=False, mask=mask)

# Show the plot
plt.xlabel('Number of use-cases requiring a DPIA by the EU/EEA Member State (using ISO 3166-2 code)')
plt.ylabel('Annex III Clause')
# plt.title("Variation across EU/EEA member states in DPIA required conditions for AI Act's Annex III clauses")
plt.savefig("anx3dpia.png", dpi=300, transparent=True, bbox_inches='tight')
# plt.show()
