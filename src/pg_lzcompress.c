#include <stdio.h>
#include <arpa/inet.h>
typedef signed int int32;
typedef unsigned int uint32;
/*
 * These structs describe the header of a varlena object that may have been
 * TOASTed.  Generally, don't reference these structs directly, but use the
 * macros below.
 *
 * We use separate structs for the aligned and unaligned cases because the
 * compiler might otherwise think it could generate code that assumes
 * alignment while touching fields of a 1-byte-header varlena.
 */
typedef union
{
	struct						/* Normal varlena (4-byte length) */
	{
		uint32		va_header;
		char		va_data[1];
	}			va_4byte;
	struct						/* Compressed-in-line format */
	{
		uint32		va_header;
		uint32		va_rawsize; /* Original data size (excludes header) */
		char		va_data[1]; /* Compressed data */
	}			va_compressed;
} varattrib_4b;
/* ----------
 * PGLZ_Header -
 *
 *		The information at the start of the compressed data.
 * ----------
 */
typedef struct PGLZ_Header
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int32		rawsize;
} PGLZ_Header;

#define VARSIZE_4B(PTR) \
	(ntohl(((varattrib_4b *) (PTR))->va_4byte.va_header) & 0x3FFFFFFF)
#define VARSIZE(PTR)						VARSIZE_4B(PTR)
/* ----------
 * pglz_decompress -
 *
 *		Decompresses source into dest.
 * ----------
 */
void
pglz_decompress(const PGLZ_Header *source, char *dest)
{
	const unsigned char *sp;
	const unsigned char *srcend;
	unsigned char *dp;
	unsigned char *destend;

	sp = ((const unsigned char *) source) + sizeof(PGLZ_Header);
	srcend = ((const unsigned char *) source) + VARSIZE(source);
	dp = (unsigned char *) dest;
	destend = dp + source->rawsize;

	while (sp < srcend && dp < destend)
	{
		/*
		 * Read one control byte and process the next 8 items (or as many as
		 * remain in the compressed input).
		 */
		unsigned char ctrl = *sp++;
		int			ctrlc;

		for (ctrlc = 0; ctrlc < 8 && sp < srcend; ctrlc++)
		{
			if (ctrl & 1)
			{
				/*
				 * Otherwise it contains the match length minus 3 and the
				 * upper 4 bits of the offset. The next following byte
				 * contains the lower 8 bits of the offset. If the length is
				 * coded as 18, another extension tag byte tells how much
				 * longer the match really was (0-255).
				 */
				int32		len;
				int32		off;

				len = (sp[0] & 0x0f) + 3;
				off = ((sp[0] & 0xf0) << 4) | sp[1];
				sp += 2;
				if (len == 18)
					len += *sp++;

				/*
				 * Check for output buffer overrun, to ensure we don't clobber
				 * memory in case of corrupt input.  Note: we must advance dp
				 * here to ensure the error is detected below the loop.  We
				 * don't simply put the elog inside the loop since that will
				 * probably interfere with optimization.
				 */
				if (dp + len > destend)
				{
					dp += len;
					break;
				}

				/*
				 * Now we copy the bytes specified by the tag from OUTPUT to
				 * OUTPUT. It is dangerous and platform dependent to use
				 * memcpy() here, because the copied areas could overlap
				 * extremely!
				 */
				while (len--)
				{
					*dp = dp[-off];
					dp++;
				}
			}
			else
			{
				/*
				 * An unset control bit means LITERAL BYTE. So we just copy
				 * one from INPUT to OUTPUT.
				 */
				if (dp >= destend)		/* check for buffer overrun */
					break;		/* do not clobber memory */

				*dp++ = *sp++;
			}

			/*
			 * Advance the control bit
			 */
			ctrl >>= 1;
		}
	}

	/*
	 * Check we decompressed the right amount.
	 */
	if (dp != destend || sp != srcend)
        printf("compressed data is corrupt"); 
	/*
	 * That's it.
	 */
}