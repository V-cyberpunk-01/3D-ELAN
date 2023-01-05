import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymatgen.core import Structure
import math
import os
from typing import List, Tuple, Any, Dict, Union, Optional
import logging
import pandas as pd
import copy
import argparse

logging.basicConfig(level=logging.INFO, handlers=None)


def arg_parser():
    parser = argparse.ArgumentParser(description='XML_Filler')
    parser.add_argument('-f',
                        '--file',
                        type=str,
                        required=True,
                        default='./cif/Nb5Si3.cif',
                        dest="filepath",
                        help='please give a cif file path')
    return parser.parse_args()


def create_logger():
    log = logging.getLogger(name="CystalShell_logger")
    console = logging.StreamHandler()
    fmt = logging.Formatter(
        fmt=
        "%(asctime)s - [%(funcName)s-->line:%(lineno)d] - %(levelname)s:%(message)s"
    )
    console.setFormatter(fmt=fmt)
    # log.handlers = None
    log.addHandler(console)
    log.propagate = False
    return log


class CrystalShell():

    def __init__(self, filename: str, center: Union[int, List[float]] = 0):
        """晶体壳层分析工具，输入若干个cif文件，找出若干壳层信息，并以图片和坐标的形式返回。
        
        Args:
            filename (str): 一个cif文件名。
        """
        self.filename = filename
        self.pos_abc: Dict = {}  #归一化后的比例
        self.pos_xyz: Dict = {}  #真实的原子坐标
        self.atoms: Dict = {}  #原子

        self.center = center  #中心原子index,默认0
        self.info: Dict = {}  #晶胞信息
        self.log = create_logger()
        self.read_cif()
        self.draw(hull=None,
                  pos=self.pos_matrix,
                  remain_convex=self.pos_matrix)
        # self.pos = self.period_extend()  #延拓后的坐标
        self.pos = self.pos_matrix
        self._cal_hist()
        # self.find_convex()
        self.shell_pos = self.find_convex()

    def read_cif(self):
        """读取cif文件。
        """
        assert os.path.isfile(self.filename), '文件或路径不存在'
        stru = Structure.from_file(self.filename)  #从文件中读出坐标
        self.info = stru.as_dict()
        for i, sp in enumerate(stru.sites):
            self.atoms.setdefault(i, str(sp.specie.symbol))
            self.pos_abc.setdefault(i, [sp.a, sp.b, sp.c])
            self.pos_xyz.setdefault(i, [sp.x, sp.y, sp.z])
            # print('读取到cif文件{}的构成为:{}'.format(self.filename, stru.formula))
        self.log.info(msg='读取到cif文件{}的构成为:{}'.format(self.filename,
                                                     stru.formula),
                      stack_info=0)
        self.pos_matrix = np.array(list(self.pos_abc.values()))
        print(self.pos_matrix.shape)
        if isinstance(self.center, List):  #如果人为指定了坐标的位置，则x
            self.center = self._cal_center()

    def period_extend(self):
        """xyz表示的是要对原晶胞做的坐标偏移、实现周期性延拓。这种拓展称为“魔方”延拓。
        returns List[List]
        """
        xy = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [1, 1, 0],
              [1, -1, 0], [-1, -1, 0], [-1, 1, 0]]  #1个中心水平方向上8个方向拓展
        self.pos_move = []
        for i in xy:
            self.pos_move.extend([i, i[:2] + [1], i[:2] + [-1]])  #向上，向下
        self.pos_move.extend([[0, 0, 1], [0, 0, -1]])
        all_pos = copy.deepcopy(self.pos_matrix)
        move_class = [
            0
        ] * self.pos_matrix.shape[0]  #move_class是为了届时还原被延拓的坐标到底是源自哪一个？
        for i in range(len(self.pos_move)):
            moved = self.calArray2dDiff(self.pos_matrix + self.pos_move[i],
                                        all_pos)
            move_class.extend([i + 1] * len(moved))
            all_pos = np.vstack((all_pos, moved))
        extend_unique_pos = np.append(all_pos,
                                      np.array(move_class).reshape(-1, 1),
                                      axis=1)
        if False:  #默认不存储延拓后的所有坐标
            np.savetxt('extend_pos.csv',
                       extend_unique_pos,
                       delimiter=",",
                       fmt='%.4f')
        return extend_unique_pos

    def calArray2dDiff(self, array_0, array_1):
        """计算两个2为列表的差。
        Args:
            array_0 (_type_): 坐标列表
            array_1 (_type_): 被减去坐标列表

        Returns:
            List[List]: 坐标差集。
        """
        array_0_rows = array_0.view([('', array_0.dtype)] * array_0.shape[1])
        array_1_rows = array_1.view([('', array_1.dtype)] * array_1.shape[1])
        return np.setdiff1d(array_0_rows,
                            array_1_rows).view(array_0.dtype).reshape(
                                -1, array_0.shape[1])

    def _cal_center(self):
        """计算中心原子

        Returns:
            int:中心原子的索引。
        """
        center_vec = np.array([
            self.info['lattice']['a'] / 2, self.info['lattice']['b'] / 2,
            self.info['lattice']['c'] / 2
        ])
        dis = np.linalg.norm(center_vec - self.pos_matrix, axis=1)
        return np.argmin(dis)

    def _cal_hist(self):
        """计算距离直方图
        
        Returns:
              None
        """
        self.dis = {}
        # print(f'center:{self.center}')
        for i in range(self.pos.shape[0]):
            if i != self.center:
                vec = self.pos[i] - self.pos[self.center]
                dis = np.around(np.linalg.norm(vec), 6)
                self.dis[i] = dis
        distances = list(self.dis.values())
        distances /= min(distances)
        self._draw_hist(distances)
        self.mid_gaps = self._get_gaps(sorted(list(set(distances))))

    def _draw_hist(self, distances):
        """绘制距离直方图

        Args:
            distances (List): 距离列表
        """
        plt.hist(x=distances,
                 bins=30,
                 range=(1, math.ceil(max(distances))),
                 cumulative=False,
                 rwidth=0.5,
                 histtype='barstacked')
        plt.xlabel('d/dmin')
        plt.ylabel('n')
        plt.show()

    def _get_gaps(self, borders):
        gaps = []
        for i in range(len(borders) - 1):
            gaps.append([round(borders[i + 1] - borders[i], 6)] +
                        list(np.array(borders[i:i + 2]) * min(borders)))
        return sorted(gaps, key=lambda x: x[0], reverse=True)[:10]

    def _get_mid_gaps(self):
        """找到所有的中间gap。
        """
        unique_d = list(set(self.dis.values()))
        self.mid_gaps = []
        for i in range(len(unique_d) - 1):
            self.mid_gaps.append(round(sum(unique_d[i:i + 2]) / 2, 6))
        self.mid_gaps.append(unique_d[-1] + 1)

    def _get_convex(self, pos):
        """找到凸包壳层。

        Args:
            pos (NDArray): 点的坐标

        Returns:
            None | hull: 一个凸包或无。
        """
        print(pos.shape)
        hull = ConvexHull(pos)
        if len(hull.vertices) != pos.shape[0]:
            return None
        else:
            if self.check_inner(hull, pos,
                                center=self.pos_matrix[self.center]):
                return hull
            else:
                return None

    def draw(self,
             hull,
             pos,
             remain_convex=None,
             real_idx=None,
             c='k',
             saveas='./shell.png'):
        """绘制壳层示意图

        Args:
            hull (_type_): 壳层凸包
            idx (_type_): 壳层原子的索引
            remain_idx (_type_): 非壳层原子的索引
            c (_type_): 颜色
            saveas (str, optional): 存储路径. Defaults to './'.
        """
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        # print(hull.vertices)
        if hull:
            convex = pos[hull.vertices, :]
        else:
            convex = remain_convex
        # print(np.setdiff1d(np.array(range(pos.shape[0])),hull.vertices))
        # ax = plt.subplot(projection = '3d')  # 创建一个三维的绘图工程
        ax.set_title('3d_convex_hull')  # 设置本图名称
        for i in range(len(convex)):
            ax.scatter(convex[i, 0],
                       convex[i, 1],
                       convex[i, 2],
                       c='r',
                       cmap='grays')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色

        ax.scatter(remain_convex[:, 0],
                   remain_convex[:, 1],
                   remain_convex[:, 2],
                   c='gray',
                   cmap='grays')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
        # ax.plot_trisurf(convex[:,0], convex[:,1], convex[:,2],cmap='viridis')
        ax.set_xlabel('X')  # 设置x坐标轴
        ax.set_ylabel('Y')  # 设置y坐标轴
        ax.set_zlabel('Z')  # 设置z坐标轴
        lines = set()
        if hull:
            for simple in hull.simplices:
                for src, dst in combinations(simple, 2):
                    # print({(src,dst)})
                    vec = pos[src, :3] - pos[dst, :3]
                    # print(np.count_nonzero(vec))
                    # if np.count_nonzero(vec)!=1:
                    #     continue
                    lines |= {(src, dst)}
        # print(lines)
        for line in lines:
            ax.plot3D(pos[line, 0], pos[line, 1], pos[line, 2], c=c)
        plt.show()
        plt.savefig(saveas)
        plt.pause(1)
        plt.close()

    def check_inner(self, hull, pos, center=[0.5, 0.5, 0.5]):
        """检查中心原子是否在壳层内部。

        Args:
            hull (_type_): 壳层凸包
            pos (_type_): 点的坐标
            center (list, optional): 中心原子坐标. Defaults to [0.5, 0.5, 0.5].

        Returns:
            Boolean:True | False
        """
        v = 0.0
        for pointidxs in hull.simplices:
            points = pos[pointidxs]
            ab = points[1] - points[0]
            ac = points[2] - points[0]
            ap = center - points[0]
            v += np.abs(np.dot(ap, np.cross(ab, ac)) / 6)  #提及相加相等也可以或说明点在内部
        if np.allclose([v], [hull.volume]):
            return True
        else:
            return False

    def which_atom_to_Change(self, vec, pos_m):
        '''
      找出相似度最高的向量,但距离最好控制在一定范围内
      '''
        dis = np.linalg.norm(vec - pos_m, axis=1)
        idx = np.argmin(dis)
        if dis[idx] < 1e-1:
            return idx
        else:
            return -1

    def find_convex(self, shell_limit=1):
        """根据cif文件找壳层信息。
        """
        mid_gaps = self.mid_gaps
        d = self.dis
        pos = self.pos
        times = min(d.values())
        colormap = [
            'r', 'orange', 'y', 'g', 'b', 'm', 'c', 'darkgrey', 'grey',
            'dimgrey', 'black'
        ]
        for i in range(len(mid_gaps)):  #目前只找第一壳层
            self.log.info(
                msg='gap:{:.3f},left_border:{:.3f},right_border:{:.3f}'.format(
                    mid_gaps[i][0], mid_gaps[i][1] * times,
                    mid_gaps[i][2] * times),
                stack_info=0)
            idx = []
            back_index = []
            for k, v in d.items():
                if v < sum(mid_gaps[i][1:3]) / 2 * times:
                    idx.append(k)
            # try:
            # print(d, idx, pos[idx, :3].shape)
            hull = self._get_convex(pos[idx, :3])
            # except:
            #     hull = None
            if hull:
                for p in pos[idx]:
                    if p[-1] != 0:
                        p = p[:3] - self.pos_move[int(p[-1] - 1)]
                    # back_pos.append(pos[:3])
                    back_index.append(
                        self.which_atom_to_Change(
                            p[:3], pos[:self.pos_matrix.shape[0], :3]))
                shell_limit -= 1
                for id in idx:
                    del d[id]
                remain_convex = pos[np.setdiff1d(np.array(range(pos.shape[0])),
                                                 np.array(idx)), :3]
                self.draw(hull,
                          pos[idx, :3],
                          remain_convex,
                          real_idx=back_index,
                          c=colormap.pop(0))
                self.log.info(msg='idx:{},back_idx:{}'.format(idx, back_index),
                              stack_info=0)
                return pos[back_index]
            if shell_limit == 0:
                break

            return None


if __name__ == '__main__':
    arg = arg_parser()
    print(arg.filepath)
    c = CrystalShell(filename=arg.filepath)
    print(c.shell_pos)