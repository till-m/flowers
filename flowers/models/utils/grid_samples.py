import torch
from torch.nn import functional as F


@torch.compile
def custom_grid_sample2d(
    input, grid, mode="bilinear", padding_modes=("zeros", "zeros"), align_corners=False
):
    """
    Optimized custom grid_sample with periodic boundary conditions.

    Instead of tiling 3x3, this pads minimally: one wrap-around row on bottom,
    one wrap-around column on right, and the corner pixel. Uses modulo arithmetic
    to handle periodic wrapping with much lower memory overhead.

    Args:
        input: Input tensor of shape (N, C, H, W)
        grid: Grid tensor of shape (N, H, W, 2) with normalized coordinates in [-1, 1]
        mode: Interpolation mode ('bilinear', 'nearest')
        padding_modes: Either a string ('zeros', 'border', 'reflection', 'periodic') or
                      a tuple (padding_h, padding_w) where each is one of those strings.
                      If tuple, at least one should be 'periodic'.
        align_corners: Whether to align corners

    Returns:
        Sampled tensor of shape (N, C, H, W)
    """

    padding_h, padding_w = padding_modes

    # If neither is periodic, just use standard grid_sample
    if padding_w.lower() != "periodic" and padding_h.lower() != "periodic":
        if padding_w != padding_h:
            msg = "When using different paddings for x and y dimensions, one should be `periodic`."
            raise ValueError(msg)
        return F.grid_sample(
            input, grid, mode=mode, padding_mode=padding_w, align_corners=align_corners
        )

    # At least one is periodic
    periodic_w = padding_w == "periodic"
    periodic_h = padding_h == "periodic"
    final_padding_w = "border" if periodic_w else padding_w
    final_padding_h = "border" if periodic_h else padding_h

    N, C, H, W = input.shape

    # Pad based on which dimensions are periodic
    if periodic_h and periodic_w:
        # Both periodic: pad both dimensions
        top_row = input[:, :, 0:1, :]  # (N, C, 1, W)
        left_col = input[:, :, :, 0:1]  # (N, C, H, 1)
        top_left_corner = input[:, :, 0:1, 0:1]  # (N, C, 1, 1)

        # Add bottom row
        padded_h = torch.cat([input, top_row], dim=2)  # (N, C, H+1, W)

        # Add right column (including the corner)
        left_col_extended = torch.cat(
            [left_col, top_left_corner], dim=2
        )  # (N, C, H+1, 1)
        padded_input = torch.cat(
            [padded_h, left_col_extended], dim=3
        )  # (N, C, H+1, W+1)

        H_padded = H + 1
        W_padded = W + 1
    elif periodic_h:
        # Only Y periodic: pad height dimension only
        top_row = input[:, :, 0:1, :]  # (N, C, 1, W)
        padded_input = torch.cat([input, top_row], dim=2)  # (N, C, H+1, W)

        H_padded = H + 1
        W_padded = W
    elif periodic_w:
        # Only X periodic: pad width dimension only
        left_col = input[:, :, :, 0:1]  # (N, C, H, 1)
        padded_input = torch.cat([input, left_col], dim=3)  # (N, C, H, W+1)

        H_padded = H
        W_padded = W + 1

    # Now map coordinates. The original domain [0, H-1] x [0, W-1] in the padded space
    # is still indexed as [0, H-1] x [0, W-1], but we have extra pixels at H and W
    # for wrap-around interpolation.

    # Convert normalized grid coordinates [-1, 1] to pixel coordinates in original space
    gx, gy = grid[..., 0], grid[..., 1]  # Each is (N, H_out, W_out)

    if align_corners:
        # Map from [-1, 1] to [0, W-1] and [0, H-1]
        px = (gx + 1) * (W - 1) / 2
        py = (gy + 1) * (H - 1) / 2

        # Apply modulo only to periodic dimensions
        if periodic_w:
            px_wrapped = px % W
        else:
            px_wrapped = px

        if periodic_h:
            py_wrapped = py % H
        else:
            py_wrapped = py

        # Convert back to normalized coordinates in the padded space
        if periodic_w:
            gx_new = 2 * px_wrapped / W - 1
        else:
            # Map to padded space (which is same size if not periodic)
            gx_new = 2 * px_wrapped / (W_padded - 1) - 1 if W_padded > 1 else gx

        if periodic_h:
            gy_new = 2 * py_wrapped / H - 1
        else:
            # Map to padded space (which is same size if not periodic)
            gy_new = 2 * py_wrapped / (H_padded - 1) - 1 if H_padded > 1 else gy
    else:
        # Map from [-1, 1] to pixel center coordinates
        px = ((gx + 1) * W - 1) / 2
        py = ((gy + 1) * H - 1) / 2

        # Apply modulo only to periodic dimensions
        if periodic_w:
            px_wrapped = px % W
        else:
            px_wrapped = px

        if periodic_h:
            py_wrapped = py % H
        else:
            py_wrapped = py

        # Convert back to normalized coordinates in the padded space
        if periodic_w:
            gx_new = 2 * (px_wrapped + 0.5) / (W + 1) - 1
        else:
            gx_new = 2 * (px_wrapped + 0.5) / W_padded - 1 if W_padded > 0 else gx

        if periodic_h:
            gy_new = 2 * (py_wrapped + 0.5) / (H + 1) - 1
        else:
            gy_new = 2 * (py_wrapped + 0.5) / H_padded - 1 if H_padded > 0 else gy

    grid_new = torch.stack([gx_new, gy_new], dim=-1)

    # Use the appropriate padding mode on the padded input
    # Note: F.grid_sample doesn't support per-dimension padding, so we use border
    # for the periodic dimensions (since we've already handled wrapping) and need
    # to use the requested mode for non-periodic dimensions
    # Since we can't mix, we use border for periodic dims and the requested mode otherwise
    if periodic_w and periodic_h:
        # Both periodic, already wrapped - use border
        final_padding = "border"
    elif periodic_w:
        # X periodic, Y not - use the Y padding mode
        final_padding = final_padding_h
    elif periodic_h:
        # Y periodic, X not - use the X padding mode
        final_padding = final_padding_w

    return F.grid_sample(
        padded_input,
        grid_new,
        mode=mode,
        padding_mode=final_padding,
        align_corners=align_corners,
    )


@torch.compile
def custom_grid_sample3d(input,
                          grid,
                          mode="bilinear",
                          padding_modes=("periodic", "periodic", "periodic"),
                          align_corners=True):
    """
    3D grid_sample with support for per-dimension periodic boundary conditions.

    Args:
        input: (N, C, D, H, W)
        grid:  (N, D_out, H_out, W_out, 3), normalized coords in [-1, 1]
               grid[..., 0] -> x (width), grid[..., 1] -> y (height), grid[..., 2] -> z (depth)
        mode: 'bilinear' (trilinear) or 'nearest'
        padding_modes: string or 3-tuple. Each entry in
                       {'zeros', 'border', 'reflection', 'periodic'}.
        align_corners: as in torch.nn.functional.grid_sample
    """

    padding_d, padding_h, padding_w = padding_modes

    periodic_d = (padding_d == "periodic")
    periodic_h = (padding_h == "periodic")
    periodic_w = (padding_w == "periodic")

    # if no dimension is periodic, fall back to standard grid_sample
    if not (periodic_d or periodic_h or periodic_w):
        # all non-periodic paddings must match
        if not (padding_d == padding_h == padding_w):
            raise ValueError(
                "When using different paddings for D, H, W and none is 'periodic', "
                "they must all be equal."
            )
        return F.grid_sample(
            input,
            grid,
            mode=mode,
            padding_mode=padding_d,
            align_corners=align_corners,
        )

    # at least one dimension is periodic
    # decide the single padding_mode we will pass to F.grid_sample
    non_periodic_paddings = []
    if not periodic_d:
        non_periodic_paddings.append(padding_d)
    if not periodic_h:
        non_periodic_paddings.append(padding_h)
    if not periodic_w:
        non_periodic_paddings.append(padding_w)

    if len(non_periodic_paddings) == 0:
        # all three periodic: once we wrap coordinates, border padding is safe
        final_padding = "border"
    else:
        # all non-periodic paddings must be identical
        if len(set(non_periodic_paddings)) > 1:
            raise ValueError(
                "Non-periodic dimensions must use the same padding_mode, "
                f"got: {set(non_periodic_paddings)}"
            )
        final_padding = non_periodic_paddings[0]

    N, C, D, H, W = input.shape

    # build minimally padded tensor: add a single wrap-around slice per periodic dim
    padded_input = input
    D_padded, H_padded, W_padded = D, H, W

    if periodic_d:
        front_slice = padded_input[:, :, 0:1, :, :]   # first depth slice;  (N, C, 1, H, W)
        padded_input = torch.cat([padded_input, front_slice], dim=2)  # (N, C, D+1, H, W)
        D_padded += 1

    if periodic_h:
        top_slice = padded_input[:, :, :, 0:1, :]     # first row along height; (N, C, D+1, 1, W)
        padded_input = torch.cat([padded_input, top_slice], dim=3) # (N, C, D+1, H+1, W)
        H_padded += 1

    if periodic_w:
        left_slice = padded_input[:, :, :, :, 0:1]    # first column along width; (N, C, D+1, H+1, 1)
        padded_input = torch.cat([padded_input, left_slice], dim=4)  # (N, C, D+1, H+1, W+1)
        W_padded += 1

    # map normalized grid -> pixel coords -> apply modulo -> back to normalized
    # grid[..., 0] is x (width), grid[..., 1] is y (height), grid[..., 2] is z (depth)
    gx = grid[..., 0]  # x / width ;  # (N, D_out, H_out, W_out)
    gy = grid[..., 1]  # y / height
    gz = grid[..., 2]  # z / depth

    if align_corners:
        # map from [-1, 1] to [0, size-1] in original space
        px = (gx + 1.0) * (W - 1.0) / 2.0
        py = (gy + 1.0) * (H - 1.0) / 2.0
        pz = (gz + 1.0) * (D - 1.0) / 2.0

        # apply modulo only to periodic dimensions
        px_wrapped = px % W if periodic_w else px
        py_wrapped = py % H if periodic_h else py
        pz_wrapped = pz % D if periodic_d else pz

        # Convert back to normalized coordinates for the padded space
        # Padded sizes are (D+1, H+1, W+1), so W_padded-1 = W, etc.
        if periodic_w:
            gx_new = 2.0 * px_wrapped / W - 1.0  # = 2*px_wrapped/W - 1
        else:
            gx_new = 2.0 * px_wrapped / (W_padded - 1.0) - 1.0 if W_padded > 1 else gx

        if periodic_h:
            gy_new = 2.0 * py_wrapped / H - 1.0
        else:
            gy_new = 2.0 * py_wrapped / (H_padded - 1.0) - 1.0 if H_padded > 1 else gy

        if periodic_d:
            gz_new = 2.0 * pz_wrapped / D - 1.0
        else:
            gz_new = 2.0 * pz_wrapped / (D_padded - 1.0) - 1.0 if D_padded > 1 else gz

    else:
        # map from [-1, 1] to pixel centers in original space
        px = ((gx + 1.0) * W - 1.0) / 2.0    # in [-0.5, W - 0.5]
        py = ((gy + 1.0) * H - 1.0) / 2.0
        pz = ((gz + 1.0) * D - 1.0) / 2.0

        # apply modulo only to periodic dimensions
        px_wrapped = px % W if periodic_w else px
        py_wrapped = py % H if periodic_h else py
        pz_wrapped = pz % D if periodic_d else pz

        # convert back to normalized coordinates in the padded space
        if periodic_w:
            gx_new = 2.0 * (px_wrapped + 0.5) / (W + 1.0) - 1.0
        else:
            gx_new = 2.0 * (px_wrapped + 0.5) / W_padded - 1.0 if W_padded > 0 else gx

        if periodic_h:
            gy_new = 2.0 * (py_wrapped + 0.5) / (H + 1.0) - 1.0
        else:
            gy_new = 2.0 * (py_wrapped + 0.5) / H_padded - 1.0 if H_padded > 0 else gy

        if periodic_d:
            gz_new = 2.0 * (pz_wrapped + 0.5) / (D + 1.0) - 1.0
        else:
            gz_new = 2.0 * (pz_wrapped + 0.5) / D_padded - 1.0 if D_padded > 0 else gz

    grid_new = torch.stack([gx_new, gy_new, gz_new], dim=-1)

    return F.grid_sample(
        padded_input,
        grid_new,
        mode=mode,
        padding_mode=final_padding,
        align_corners=align_corners,
    )
