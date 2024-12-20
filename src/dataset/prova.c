glue(cirrus_bitblt_rop_fwd_, ROP_NAME) (CirrusVGAState
					* s, uint8_t * dst, const uint8_t
					* src, int dstpitch, int
					srcpitch, int bltwidth, int
					bltheight) { int x, y;
	dstpitch -= bltwidth;
	srcpitch -= bltwidth;
	for (y = 0; y < bltheight; y++) {
		for (x = 0; x < bltwidth; x++) {
			ROP_OP(*dst, *src);
			dst++;
			src++;
		}
		dst += dstpitch;
		src += srcpitch;
	}
}
