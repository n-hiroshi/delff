from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import io
from PIL import Image
import numpy as np

def draw_molecule(smiles, values):
    # 分子を作成
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # 分子の座標を計算
    rdDepictor.Compute2DCoords(mol)

    # 最大・最小の値を取得
    min_value = min(values)
    max_value = max(values)

    # 原子の値をカラーマップに変換
    atom_colors = {}
    for i, atom_value in enumerate(values):
        color = cm.get_cmap('bwr')(1 - (atom_value - min_value) / (max_value - min_value))  # カラーマップを'bwr'に変更し、色を逆にする
        atom_colors[i] = tuple(color)

    # 分子を描画
    d2d = rdMolDraw2D.MolDraw2DCairo(300, 300)  # 画像サイズを600x600に設定
    d2d_options = rdMolDraw2D.MolDrawOptions()
    d2d_options.colors = {'C': (128,128,128), 'O': (128,128,128), 'N': (128,128,128), 'H': (128,128,128)}
    #d2d.drawOptions().addAtomIndices = True
    #d2d.drawOptions().prepareMolsBeforeDrawing = False
    #d2d.drawOptions().atomLabelFontSize = 24  # 原子番号のフォントサイズを24に設定
    #d2d.drawOptions().colors = {'C':(128,128,128), 'O':(128,128,128), 'N':(128,128,128), 'H':(128,128,128)}
    d2d.DrawMolecule(mol, highlightAtoms=list(range(mol.GetNumAtoms())), highlightAtomColors=atom_colors, highlightBonds=[])
    d2d.FinishDrawing()

    # 画像を表示
    img = Image.open(io.BytesIO(d2d.GetDrawingText()))
    plt.imshow(img)
    plt.axis('off')

    # カラーバーを作成
    #reversed_bwr = ListedColormap(cm.bwr.colors[::-1])  # カラーマップ'bwr'の色を反転
    reversed_bwr = ListedColormap(cm.get_cmap('bwr')(np.linspace(0, 1, 256))[::-1])

    sm = plt.cm.ScalarMappable(cmap=reversed_bwr, norm=plt.Normalize(vmin=min_value, vmax=max_value))  # カラーバーのカラーマップを反転した'bwr'に変更
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Values')

    plt.show()

# 水素原子を明示したSMILES
smiles = "c1ccccc1NC(=O)C"

# 各原子に対応する値 (水素原子も含む)
values = [-0.014, -0.019, -0.011, -0.019, -0.014,
           0.001, -0.003, -0.005, -0.039,  0.010,
          -0.006, -0.008, -0.011, -0.008, -0.006,
           0.040,  0.035,  0.035,  0.035]

# 分子構造を表示
draw_molecule(smiles, values)

