
import numpy as np
import torch

TEMP = 0.5
np.set_printoptions(suppress=True, precision=4)



# raw_logits_arr = np.array([[-1.119, -1.1155, -1.114,  -1.1137, -1.112,  -1.1103, -1.1119, -1.1114, -1.1099,
#   -1.1086, -1.1075, -1.1062, -1.107,  -1.1058, -1.113,  -1.114,  -1.114,  -1.1136,
#   -1.1116, -1.1103, -1.1093, -1.1093, -1.1093, -1.1113, -1.1101, -1.1115, -1.1101,
#   -1.1068, -1.1168, -1.1147, -1.1143, -1.1129, -1.114,  -1.1126, -1.0409, -0.6471,
#   -1.1115, -1.1129, -1.1145, -1.114,  -1.1142, -1.1146, -1.1197, -1.1178, -1.1178,
#   -1.1178, -1.1173, -1.1158, -0.9146, -0.9849, -1.1164, -1.1177, -1.1196, -1.1191,
#   -1.1185, -1.119,  -1.121,  -1.122,  -1.121,  -1.1207, -1.1114, -1.1178, -1.1188,
#   -0.8964, -1.0352, -1.1076, -1.1213, -1.1215, -1.1212, -1.1207, -1.12,   -1.1233,
#   -1.1226, -1.1322, -1.049,  -1.1203, -1.118,  -1.0382, -0.9339, -0.8505, -1.1367,
#   -1.1211, -1.1219, -1.1204, -1.1214, -1.1222, -1.1229, -1.1486, -1.0441, -1.0982,
#   -1.1167, -1.127,  -1.025,  -0.9819, -1.0762, -1.1214, -1.1221, -1.1215, -1.1194,
#   -1.1188, -1.1183, -1.1131, -1.04,   -1.0064, -1.1202, -1.1043, -1.0058, -1.0492,
#   -1.0406, -1.1096, -1.1217, -1.0954, -1.1089, -1.107,  -1.1111, -1.1102, -1.1377,
#   -1.0541, -1.1194, -1.1171, -1.1426, -1.1327, -1.0508, -1.1212, -1.114,  -1.0998,
#   -1.1081, -1.1147, -1.1091, -1.1094, -1.0072, -1.1417, -0.9819, -1.1236, -1.1447,
#   -1.1335, -1.1401, -1.1198, -1.0985, -1.0999, -1.11,   -1.1436, -1.1362, -1.1161,
#   -0.9901, -1.1304, -1.0916, -1.0607, -1.1328, -1.0307, -1.1193, -1.1204, -1.1166,
#   -1.1091, -1.1102, -1.1175, -1.1551, -1.1418, -1.0915, -1.1391, -1.1358, -1.1249,
#   -1.1149, -1.1283, -1.1022, -1.1133, -1.1162, -1.0785, -1.1156, -1.1163, -1.1096,
#   -1.1516, -1.0602, -1.1175, -1.1354, -1.1092, -1.1224, -1.1176, -1.1183, -1.1191,
#   -1.1189, -1.1226, -1.1137, -1.1197, -1.1433, -1.1314, -1.1493, -1.0639, -1.1435,
#   -1.085,  -1.0688, -1.1229, -1.1221, -1.1168, -1.1208, -1.1365]])

raw_logits_arr = np.array([[-8, -7, -6, -5, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]])


TOY = (not raw_logits_arr.shape[0] == 196)

raw_logits = torch.tensor(raw_logits_arr)
boundary_probs = torch.sigmoid(raw_logits)
print("Boundary probabilities after sigmoid:")
print(boundary_probs.cpu().numpy())

bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
    temperature=TEMP,
    probs=boundary_probs,
)

soft_boundaries = bernoulli.rsample()
print("Boundary probabilities after sampling:")
print(soft_boundaries.cpu().numpy())

# visualize it with a probability heatmap
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 3))
plt.imshow(boundary_probs.cpu().numpy(), cmap='viridis', aspect='auto')
plt.colorbar(label='Boundary Probability')
plt.title('Boundary Probabilities Heatmap (before sampling)')
plt.xlabel('Token Index')
plt.yticks([])
if TOY:
  plt.savefig('/users/PAS2912/yusenpeng/Fast-CLIP/src/TOY_heatmap_before_sampling.png')
else:
  plt.savefig('/users/PAS2912/yusenpeng/Fast-CLIP/src/heatmap_before_sampling.png')
plt.close()

plt.figure(figsize=(10, 3))
plt.imshow(soft_boundaries.cpu().numpy(), cmap='viridis', aspect='auto')
plt.colorbar(label='Boundary Probability')
plt.title('Boundary Probabilities Heatmap (after sampling)')
plt.xlabel('Token Index')
plt.yticks([])
if TOY:
  plt.savefig('/users/PAS2912/yusenpeng/Fast-CLIP/src/TOY_heatmap_after_sampling.png')
else:
  plt.savefig('/users/PAS2912/yusenpeng/Fast-CLIP/src/heatmap_after_sampling.png')
plt.close()

