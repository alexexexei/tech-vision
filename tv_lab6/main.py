import cv2
import numpy as np


def dilate(I, ker_sz=(5, 5), iters=1):
    ker = np.ones(ker_sz, np.uint8)
    return cv2.dilate(I, ker, iterations=iters)


def erode(I, ker_sz=(5, 5), iters=1):
    ker = np.ones(ker_sz, np.uint8)
    return cv2.erode(I, ker, iterations=iters)


def opening(I, ker_sz=(5, 5)):
    ker = np.ones(ker_sz, np.uint8)
    return cv2.morphologyEx(I, cv2.MORPH_OPEN, ker)


def closing(I, ker_sz=(5, 5)):
    ker = np.ones(ker_sz, np.uint8)
    return cv2.morphologyEx(I, cv2.MORPH_CLOSE, ker)


def find_counters(I, Inew):
    contours, _ = cv2.findContours(Inew, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contour_I = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_I, contours, -1, (0, 0, 255), 2)
    return contour_I


def separate_objs(I, iters, mellipse=(5,5)):
    ret, Inew = cv2.threshold(I, 160, 255, cv2.THRESH_BINARY_INV)
    B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, mellipse)

    BW2 = cv2.morphologyEx(Inew,
                           cv2.MORPH_ERODE,
                           B,
                           iterations=iters,
                           borderType=cv2.BORDER_CONSTANT,
                           borderValue=(0))

    T = np.zeros_like(Inew)
    while cv2.countNonZero(BW2) < BW2.size:
        D = cv2.dilate(BW2, B, borderType=cv2.BORDER_CONSTANT, borderValue=(0))
        C = cv2.morphologyEx(D,
                             cv2.MORPH_CLOSE,
                             B,
                             borderType=cv2.BORDER_CONSTANT,
                             borderValue=(0))
        S = C - D
        T = cv2.bitwise_or(S, T)
        BW2 = D

    T = cv2.morphologyEx(T,
                         cv2.MORPH_CLOSE,
                         B,
                         iterations=iters,
                         borderType=cv2.BORDER_CONSTANT,
                         borderValue=(255))

    Inew = cv2.bitwise_and(~T, Inew)
    return Inew


def bwareaopen(A, dim, conn=8):
    if A.ndim > 2:
        return None

    num, labels, stats, centers = \
        cv2.connectedComponentsWithStats(A,
            connectivity = conn)

    for i in range(num):
        if stats[i, cv2.CC_STAT_AREA] < dim:
            A[labels == i] = 0
    return A


def segmentation(I, dim, conn=8, mellpise=(5, 5), dis=5, coeff=0.6, color=(255, 0, 0)):
    I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    ret, I_bw = cv2.threshold(I_gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    I_bw = bwareaopen(I_bw, dim, conn)
    B = cv2.getStructuringElement (\
        cv2.MORPH_ELLIPSE, mellpise)
    I_bw = cv2.morphologyEx (I_bw,\
            cv2.MORPH_CLOSE, B)

    I_fg = cv2.distanceTransform(I_bw, cv2.DIST_L2, dis)
    ret, I_fg = cv2.threshold(I_fg, coeff * I_fg.max(), 255, 0)
    I_fg = I_fg.astype(np.uint8)
    ret, markers = cv2.connectedComponents(I_fg)

    I_bg = np.zeros_like(I_bw)
    markers_bg = markers.copy()
    markers_bg = cv2.watershed(I, markers_bg)
    I_bg[markers_bg == -1] = 255

    I_unk = cv2.subtract(~I_bg, I_fg)

    markers = markers + 1
    markers[I_unk == 255] = 0

    markers = cv2.watershed(I, markers)
    markers_jet = cv2.applyColorMap(
        (markers.astype(np.float32) * 255 / (ret + 1)).astype(np.uint8),
        cv2.COLORMAP_JET)
    I[markers == -1] = color
    return I, markers_jet


if __name__ == '__main__':
    bm = cv2.imread("source/base_morf.png", cv2.IMREAD_COLOR)

    dil = dilate(bm, iters=3)
    er = erode(bm, iters=4)
    op = opening(bm, ker_sz=(15, 15))
    cl = closing(bm, ker_sz=(18, 18))
    cv2.imwrite("renders/dil_bm.png", dil)
    cv2.imwrite("renders/er_bm.png", er)
    cv2.imwrite("renders/op_bm.png", op)
    cv2.imwrite("renders/cl_bm.png", cl)

    binary_I = cv2.imread("source/bin.png", cv2.IMREAD_GRAYSCALE)

    binary_Inew = separate_objs(binary_I, 10)
    c_im = find_counters(binary_I, binary_Inew)
    cv2.imwrite(f"renders/bin_new.png", binary_Inew)
    cv2.imwrite(f"renders/bin_new_c.png", c_im)

    seg_I = cv2.imread("source/seg.jpg", cv2.IMREAD_COLOR)
    
    seg_Inew, smj = segmentation(seg_I, dim=20, conn=4, mellpise=(5, 5), coeff=0.6)
    cv2.imwrite(f"renders/seg_new.jpg", seg_Inew)
    cv2.imwrite(f"renders/seg_new_mj.jpg", smj)
