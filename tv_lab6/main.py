import cv2
import numpy as np


def make_bin(I_bgr, par1, par2, par3):
    if len(I_bgr.shape) == 3 and I_bgr.shape[2] == 3:
        I_gray = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2GRAY)
    else:
        I_gray = I_bgr
    ret, I_bin = cv2.threshold(I_gray, par1, par2, par3)
    return ret, I_bin


def dilate(I, ker_sz=(5, 5), iters=1, bgr=False):
    I_bin = I
    if not bgr:
        _, I_bin = make_bin(I, 160, 255, cv2.THRESH_BINARY)
    ker = np.ones(ker_sz, np.uint8)
    return cv2.dilate(I_bin, ker, iterations=iters)


def erode(I, ker_sz=(5, 5), iters=1, bgr=False):
    I_bin = I
    if not bgr:
        _, I_bin = make_bin(I, 160, 255, cv2.THRESH_BINARY)
    ker = np.ones(ker_sz, np.uint8)
    return cv2.erode(I_bin, ker, iterations=iters)


def opening(I, ker_sz=(5, 5), bgr=False):
    I_bin = I
    if not bgr:
        _, I_bin = make_bin(I, 160, 255, cv2.THRESH_BINARY)
    ker = np.ones(ker_sz, np.uint8)
    return cv2.morphologyEx(I_bin, cv2.MORPH_OPEN, ker)


def closing(I, ker_sz=(5, 5), bgr=False):
    I_bin = I
    if not bgr:
        _, I_bin = make_bin(I, 160, 255, cv2.THRESH_BINARY)
    ker = np.ones(ker_sz, np.uint8)
    return cv2.morphologyEx(I_bin, cv2.MORPH_CLOSE, ker)


def find_counters(Inew, col=(0, 0, 255), th=2):
    contours, _ = cv2.findContours(Inew, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contour_I = cv2.cvtColor(Inew, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_I, contours, -1, col, th)
    return contour_I


def inner_contour(A):
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion = cv2.erode(A, B)
    inner_contour = A - erosion
    return inner_contour


def outer_contour(A):
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(A, B)
    outer_contour = dilation - A
    return outer_contour


def separate_objs(I, iters, mellipse=(5,5)):
    ret, Inew = make_bin(I, 160, 255, cv2.THRESH_BINARY_INV)
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
    ret, I_bw = make_bin(I, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
    path = 'tv_lab6'
    src = 'source'
    render = 'renders'
    
    bm = cv2.imread(f'{path}/{src}/base_morf.png', cv2.IMREAD_COLOR)

    col = True
    dil = dilate(bm, iters=3, bgr=col)
    er = erode(bm, iters=4, bgr=col)
    op = opening(bm, ker_sz=(15, 15), bgr=col)
    cl = closing(bm, ker_sz=(18, 18), bgr=col)
    cv2.imwrite(f"{path}/{render}/dil_bm.png", dil)
    cv2.imwrite(f"{path}/{render}/er_bm.png", er)
    cv2.imwrite(f"{path}/{render}/op_bm.png", op)
    cv2.imwrite(f"{path}/{render}/cl_bm.png", cl)

    ans = dilate(erode(bm, iters=4, bgr=col), iters=9, bgr=col)
    ans2 = closing(opening(bm, (15, 15), bgr=col), (18, 18), bgr=col)
    cv2.imwrite(f"{path}/{render}/er_then_dil_bm.png", ans)
    cv2.imwrite(f"{path}/{render}/op_then_cl_bm.png", ans2)

    bin_I = cv2.imread(f"{path}/{src}/bin.png", cv2.IMREAD_COLOR)

    bin_Inew = separate_objs(bin_I, 10)
    c_im = find_counters(bin_Inew)
    c_im2 = outer_contour(bin_Inew)
    cv2.imwrite(f"{path}/{render}/bin_new.png", bin_Inew)
    cv2.imwrite(f"{path}/{render}/bin_new_c.png", c_im)
    cv2.imwrite(f"{path}/{render}/bin_new_c2.png", c_im2)

    seg_I = cv2.imread(f"{path}/{src}/seg.jpg", cv2.IMREAD_COLOR)
    
    seg_Inew, smj = segmentation(seg_I, dim=20, conn=4, mellpise=(5, 5), coeff=0.6)
    cv2.imwrite(f"{path}/{render}/seg_new.jpg", seg_Inew)
    cv2.imwrite(f"{path}/{render}/seg_new_mj.jpg", smj)
