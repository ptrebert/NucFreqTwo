#!/usr/bin/env python3
import argparse
import pathlib as pl
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd
import pysam
import seaborn as sns

matplotlib.use("agg")

__author__ = "Mitchell Vollger"
__developer__ = "Peter Ebert"


def parse_command_line():

    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    io_group = parser.add_argument_group("File I/O")

    io_group.add_argument(
        "--infile",
        "-i",
        type=lambda x: pl.Path(x).resolve(strict=True),
        dest="infile",
        help="Path to BAM input (mode: process) or hdf (mode: plot).",
        required=True
    )
    io_group.add_argument(
        "--in-bed-regions"
        "--bed",
        type=lambda x: pl.Path(x).resolve(strict=True),
        dest="in_bed_regions",
        default=None,
        help="BED file with regions to plot. Default: None"
    )
    io_group.add_argument(
        "--out-bed-data",
        "--obed",
        type=lambda x: pl.Path(x).resolve(strict=True),
        dest="out_bed_data",
        default=None,
        help="Output a BED file with the data points. Default: None"
    )
    io_group.add_argument(
        "--repeatmasker",
        "-rm",
        type=lambda x: pl.Path(x).resolve(strict=True),
        default=None,
        dest="repeatmasker",
        help="Path to repeatmasker output to add to the plot. Default: None"
    )
    io_group.add_argument(
        "--out-plot",
        "-op",
        type=lambda x: pl.Path(x).resolve(strict=False),
        dest="out_plot",
        help="Path to plot output file.",
        default=None
    )

    runtime_group = parser.add_argument_group("Runtime parameters")

    runtime_group.add_argument(
        "--mode",
        "-m",
        choices=["process", "plot"],
        default=None,
        help="Specify run mode: process BAM file or plot processed data. Default: None",
        required=True
    )
    runtime_group.add_argument(
        "--threads",
        "-t",
        type=int,
        default=4,
        help="Number of threads to use for reading a BAM input file. Default: 4"
    )
    # The default False should indicate that
    # only sequence positions with coverage
    # will be part of the output (my guess).
    runtime_group.add_argument(
        "--all-positions",
        "-a",
        action="store_true",
        dest="all_positions",
        default=False,
        help="Output all positions. Default: False"
    )
    runtime_group.add_argument(
        "--regions",
        type=str,
        nargs="*",
        default=None,
        help="Subset BAM to these regions; format: (.*):(\d+)-(\d+)"
    )
    runtime_group.add_argument(
        "--out-min-discordant-bases",
        "--minobed",
        type=int,
        default=2,
        help="min number of discordant bases to report in output BED. Default: 2"
    )

    plot_group = parser.add_argument_group("Plotting options")

    plot_group.add_argument(
        "--legend",
        action="store_true",
        default=False,
        help="Place legend in plot with location 'best'. Default: False"
    )
    plot_group.add_argument(
        "--zerostart",
        action="store_true",
        default=False,
        help="Adjust x-ticks to 0-based coordinates. Default: False"
    )
    plot_group.add_argument(
        "-y", "--ylim",
        help="max y axis limit",
        type=float,
        default=None
    )
    plot_group.add_argument(
        "-f", "--font-size",
        help="plot font-size",
        type=int,
        default=16
    )
    plot_group.add_argument(
        "--freey",
        action="store_true",
        default=False
    )
    plot_group.add_argument(
        "--height",
        help="figure height",
        type=float, default=4
    )
    plot_group.add_argument(
        "-w", "--width",
        help="figure width",
        type=float,
        default=16
    )
    plot_group.add_argument(
        "--dpi",
        help="dpi for png",
        type=float,
        default=600
    )
    plot_group.add_argument(
        "--header",
        action="store_true",
        default=False
    )
    plot_group.add_argument(
        "--psvsites",
        help="CC/mi.gml.sites",
        default=None
    )
    plot_group.add_argument(
        "-s", "--soft",
        action="store_true",
        default=False
    )
    plot_group.add_argument(
        "-c",
        "--minclip",
        help="min number of clippsed bases in order to be displayed",
        type=float,
        default=1000,
    )

    # following: maybe was a debug switch?
    parser.add_argument(
        "-d",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS  # because not used
    )
    args = parser.parse_args()

    return args

M = 0  # M  BAM_CMATCH      0
I = 1  # I  BAM_CINS        1
D = 2  # D  BAM_CDEL        2
N = 3  # N  BAM_CREF_SKIP   3
S = 4  # S  BAM_CSOFT_CLIP  4
H = 5  # H  BAM_CHARD_CLIP  5
P = 6  # P  BAM_CPAD        6
E = 7  # =  BAM_CEQUAL      7
X = 8  # X  BAM_CDIFF       8
B = 9  # B  BAM_CBACK       9
NM = 10  # NM       NM tag  10
conRef = [M, D, N, E, X]  # these ones "consume" the reference
conQuery = [M, I, S, E, X]  # these ones "consume" the query
conAln = [M, I, D, N, S, E, X]  # these ones "consume" the alignments


def getSoft(read, group=0):
    """Check if beginning or end of read alignment
    is soft- or hard-masked.

    Args:
        read (_type_): _description_
        group (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    rtn = []
    cigar = read.cigartuples
    start = cigar[0]
    end = cigar[-1]
    if start[0] in [S, H]:
        rtn.append(
            (read.reference_name, "start", start[1], read.reference_start, read, group)
        )
    if end[0] in [S, H]:
        rtn.append(
            (read.reference_name, "end", end[1], read.reference_end, read, group)
        )
    return rtn


def parse_user_regions(user_regions, refs, regions):

    for region in user_regions:
        match = re.match("(.+):(\d+)-(\d+)", region)
        assert match is not None, f"Region is not valid: {region}"
        chrom, start, end = match.groups()
        # TODO this smells like a bug - looks like
        # specifying more than one region per chromosome
        # would simply overwrite the information in refs?
        refs[chrom] = [int(start), int(end)]
        regions.append((chrom, int(start), int(end)))
    return None


def parse_regions_from_bed_input(bed_input, refs, regions):

    # TODO could support compressed bed input
    with open(bed_input, "r") as bed_file:
        for ln, line in enumerate(bed_file, start=1):
            if line.startswith("#"):
                continue
            if not line.strip():
                continue
            try:
                chrom, start, end = line.strip().split()[:3]
                start = int(start)
                end = int(end)
            except (ValueError, TypeError, IndexError):
                sys.stderr.write(f"\nERROR - could not process BED line {ln}: {line.strip()}")
                raise
            else:
                # === same as above in parse_user_regions ===
                # TODO this smells like a bug - looks like
                # specifying more than one region per chromosome
                # would simply overwrite the information in refs?
                refs[chrom] = [start, end]
                regions.append((chrom, start, end))
    return None


def get_regions_from_bam(bam_file, threads, refs, regions):

    with pysam.AlignmentFile(bam_file, threads=threads)


def process_bam_file(args):

    refs = {}
    regions = []
    if args.regions is not None:
        parse_user_regions(args.regions, refs, regions)
    if args.in_bed_regions is not None:
        parse_regions_from_bed_input(args.in_bed_regions, refs, regions)

    if not regions:
        # implies that all regions from the BAM file are to be processed





soft = []
bam = pysam.AlignmentFile(args.infile, threads=args.threads)
refs = {}
regions = []
if args.regions is not None or args.bed is not None:
    sys.stderr.write("Reading in the region or bed argument(s).\n")
    if args.regions is not None:
        for region in args.regions:
            match = re.match("(.+):(\d+)-(\d+)", region)
            assert match, region + " not valid!"
            chrm, start, end = match.groups()
            refs[chrm] = [int(start), int(end)]
            regions.append((chrm, int(start), int(end)))

    if args.bed is not None:
        for line in open(args.bed):
            if line[0] == "#":
                continue
            line = line.strip().split()
            chrm, start, end = line[0:3]
            refs[chrm] = [int(start), int(end)]
            regions.append((chrm, int(start), int(end)))

else:

    # TODO
    # continue refactoring after this line

    sys.stderr.write(
        "Reading the whole bam becuase no region or bed argument was made.\n"
    )
    for read in bam.fetch(until_eof=True):
        ref = read.reference_name
        # read.query_qualities = [60] * len(read.query_sequence)
        soft += getSoft(read)
        if ref not in refs:
            if args.a:
                refs[ref] = [0, 2147483648]
            else:
                refs[ref] = [2147483648, 0]

        start = read.reference_start
        end = read.reference_end
        if refs[ref][0] > start:
            refs[ref][0] = start
        if refs[ref][1] < end:
            refs[ref][1] = end
    for contig in refs:
        regions.append((contig, refs[contig][0], refs[contig][1]))


def getCovByBase(contig, start, end):
    # coverage = bam.count_coverage(contig, quality_threshold=None, start=start, stop=end, read_callback="nofilter")
    # sys.stderr.write(f"{contig},{start},{end}\n")
    coverage = bam.count_coverage(
        contig, start=start, stop=end, read_callback="nofilter", quality_threshold=None
    )
    assert len(coverage) == 4
    cov = {}
    cov["A"] = coverage[0]
    cov["C"] = coverage[1]
    cov["T"] = coverage[2]
    cov["G"] = coverage[3]
    return cov


#
# creates a table of nucleotide frequescies
#
# nf = []
nf = {"contig": [], "position": [], "A": [], "C": [], "G": [], "T": [], "group": []}
GROUPS = 0
for contig, start, end in regions:
    # start, end = refs[contig]
    sys.stderr.write(
        "Reading in NucFreq from region: {}:{}-{}\n".format(contig, start, end)
    )
    cov = getCovByBase(contig, start, end)
    contiglen = len(cov["A"])
    if contiglen > 0:
        # for i in range(contiglen):
        #  nf.append( [contig, start + i, cov["A"][i], cov["C"][i], cov["G"][i], cov["T"][i] ]   )
        nf["contig"] += [contig] * contiglen
        nf["group"] += [GROUPS] * contiglen
        nf["position"] += list(range(start, start + contiglen))
        nf["A"] += cov["A"]
        nf["C"] += cov["C"]
        nf["G"] += cov["G"]
        nf["T"] += cov["T"]
        if args.soft:
            for read in bam.fetch(contig, start, end):
                soft += getSoft(read, group=GROUPS)
        GROUPS += 1


# df = pd.DataFrame(nf, columns=["contig", "position", "A", "C", "G", "T"])
df = pd.DataFrame(nf)
sort = np.flip(np.sort(df[["A", "C", "G", "T"]].values), 1)
df["first"] = sort[:, 0]
df["second"] = sort[:, 1]
df["third"] = sort[:, 2]
df["fourth"] = sort[:, 3]
df.sort_values(by=["contig", "position", "second"], inplace=True)


soft = pd.DataFrame(
    soft, columns=["contig", "side", "value", "position", "read", "group"]
)
# print(len(soft), file=sys.stderr)
soft = soft[soft.value >= args.minclip]
# print(len(soft), file=sys.stderr)
soft.sort_values(by=["contig", "position"], inplace=True)


RM = None
colors = sns.color_palette()
cmap = {}
counter = 0
if args.repeatmasker is not None:
    names = [
        "score",
        "perdiv",
        "perdel",
        "perins",
        "qname",
        "start",
        "end",
        "left",
        "strand",
        "repeat",
        "family",
        "rstart",
        "rend",
        "rleft",
        "ID",
    ]
    lines = []
    for idx, line in enumerate(args.repeatmasker):
        if idx > 2:
            lines.append(line.strip().split()[0:15])

    RM = pd.DataFrame(lines, columns=names)
    RM.start = RM.start.astype(int)
    RM.end = RM.end.astype(int)
    RM["label"] = RM.family.str.replace("/.*", "")
    for idx, lab in enumerate(sorted(RM.label.unique())):
        cmap[lab] = colors[counter % len(colors)]
        counter += 1
    RM["color"] = RM.label.map(cmap)

    args.repeatmasker.close()


sys.stderr.write("Plotting {} regions in {}\n".format(GROUPS, args.outfile))
# SET up the plot based on the number of regions
HEIGHT = GROUPS * args.height
# set text size
matplotlib.rcParams.update({"font.size": args.font_size})
# make axes
fig, axs = plt.subplots(nrows=GROUPS, ncols=1, figsize=(args.width, HEIGHT))
if GROUPS == 1:
    axs = [axs]
# make space for the bottom label of the plot
# fig.subplots_adjust(bottom=0.2)
# set figure YLIM
YLIM = int(max(df["first"]) * 1.05)

# iterate over regions
counter = 0
for group_id, group in df.groupby(by="group"):
    if args.freey:
        YLIM = int(max(group["first"]) * 1.05)

    contig = list(group.contig)[0]

    truepos = group.position.values
    first = group["first"].values
    second = group["second"].values

    # df = pd.DataFrame(nf, columns=["contig", "position", "A", "C", "G", "T"])
    if args.obed:
        tmp = group.loc[
            group.second >= args.minobed,
            ["contig", "position", "position", "first", "second"],
        ]
        if counter == 0:
            tmp.to_csv(
                args.obed,
                header=["#contig", "start", "end", "first", "second"],
                sep="\t",
                index=False,
            )
        else:
            tmp.to_csv(args.obed, mode="a", header=None, sep="\t", index=False)

    # get the correct axis
    ax = axs[group_id]

    if RM is not None:
        rmax = ax
        sys.stderr.write("Subsetting the repeatmakser file.\n")
        rm = RM[
            (RM.qname == contig) & (RM.start >= min(truepos)) & (RM.end <= max(truepos))
        ]
        assert len(rm.index) != 0, "No matching RM contig"
        # rmax.set_xlim(rm.start.min(), rm.end.max())
        # rmax.set_ylim(-1, 9)
        rmlength = len(rm.index) * 1.0
        rmcount = 0
        rectangles = []
        height_offset = max(YLIM, args.ylim) / 20
        for idx, row in rm.iterrows():
            rmcount += 1
            sys.stderr.write(
                "\rDrawing the {} repeatmasker rectangles:\t{:.2%}".format(
                    rmlength, rmcount / rmlength
                )
            )
            width = row.end - row.start
            rect = patches.Rectangle(
                (row.start, max(YLIM, args.ylim) - height_offset),
                width,
                height_offset,
                linewidth=1,
                edgecolor="none",
                facecolor=row.color,
                alpha=0.75,
            )
            rmax.add_patch(rect)
            # rectangles.append(rect)

        sys.stderr.write("\nPlotting the repeatmasker rectangles.\n")
        # rmax.add_collection(
        #    PatchCollection(
        #        rectangles,
        # match_original=True
        # facecolor=rm.color,
        # edgecolor='none'
        #    ))
        plt.show()
        sys.stderr.write("Done plotting the repeatmasker rectangles.\n")

    (prime,) = ax.plot(
        truepos,
        first,
        "o",
        color="black",
        markeredgewidth=0.0,
        markersize=2,
        label="most frequent base pair",
    )
    (sec,) = ax.plot(
        truepos,
        second,
        "o",
        color="red",
        markeredgewidth=0.0,
        markersize=2,
        label="second most frequent base pair",
    )

    # inter = int( (max(truepos)-min(truepos))/50)
    # sns.lineplot(  (truepos/inter).astype(int)*inter, first, ax = ax, err_style="bars")

    maxval = max(truepos)
    minval = min(truepos)
    subval = 0

    title = "{}:{}-{}\n".format(contig, minval, maxval)
    if GROUPS > 1:
        ax.set_title(title, fontweight="bold")
    sys.stderr.write(title)

    if args.zerostart:
        subval = minval - 1
        ax.set_xticks([x for x in ax.get_xticks() if (x - subval > 0) and (x < maxval)])
        maxval = maxval - minval

    if maxval < 1000000:
        xlabels = [format((label - subval), ",.0f") for label in ax.get_xticks()]
        lab = "bp"
    elif maxval < 10000000:
        xlabels = [format((label - subval) / 1000, ",.1f") for label in ax.get_xticks()]
        lab = "kbp"
    else:
        xlabels = [format((label - subval) / 1000, ",.1f") for label in ax.get_xticks()]
        lab = "kbp"
        # xlabels = [format( (label-subval)/1000000, ',.2f') for label in ax.get_xticks()]
        # lab = "Mbp"

    if args.ylim is not None:
        ax.set_ylim(0, args.ylim)
    else:
        ax.set_ylim(0, YLIM)

    ax.set_xlabel("Assembly position ({})".format(lab), fontweight="bold")
    ax.set_ylabel("Sequence read depth", fontweight="bold")

    # Including this causes some internal bug in matplotlib when the font-size changes
    # ylabels = [format(label, ",.0f") for label in ax.get_yticks()]
    # ax.set_yticklabels(ylabels)
    ax.set_xticklabels(xlabels)

    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    if counter == 0 and args.legend:
        lgnd = plt.legend(loc="upper right")
        for handle in lgnd.legendHandles:
            handle._sizes = [300.0]

    if args.soft:
        tmpsoft = soft[soft.group == group_id]
        if len(tmpsoft) > 0:
            axsoft = ax.twinx()
            axsoft.invert_yaxis()
            bins = args.width * 5
            color = "darkgreen"
            sns.distplot(
                tmpsoft.position,
                bins=bins,
                kde=False,
                ax=axsoft,
                hist_kws={
                    "weights": tmpsoft.value / 1000,
                    "alpha": 0.25,
                    "color": color,
                },
            )
            bot, top = axsoft.get_ylim()
            axsoft.set_ylim(1.1 * bot, 0)
            axsoft.set_xlim(minval, maxval)
            # Hide the right and top spines
            axsoft.spines["top"].set_visible(False)
            # Only show ticks on the left and bottom spines
            axsoft.yaxis.set_ticks_position("right")
            # axsoft.xaxis.set_ticks_position('bottom')
            axsoft.tick_params(axis="y", colors=color)
            axsoft.set_ylabel("Clipped Bases (kbp)", color=color)

    if args.psvsites is not None:  # and len(args.psvsites)>0):
        cuts = {}
        for idx, line in enumerate(open(args.psvsites).readlines()):
            try:
                vals = line.strip().split()
                cuts[idx] = list(map(int, vals))
                # make plot
                x = np.array(cuts[idx]) - 1
                idxs = np.isin(truepos, x)
                y = second[idxs]
                ax.plot(x, y, alpha=0.5)  # , label="group:{}".format(idx) )
            except Exception as e:
                print("Skipping because error: {}".format(e), file=sys.stderr)
                continue

    # outpath = os.path.abspath(args.outfile)
    # if(counter == 0):
    #  outf =   outpath
    # else:
    #  name, ext = os.path.splitext(outpath)
    #  outf = "{}_{}{}".format(name, counter + 1, ext)

    counter += 1

plt.tight_layout()
plt.savefig(args.outfile, dpi=args.dpi)


def main():

    args = parse_command_line()

    if args.mode == "process":
        # check that BAM index exists at standard path
        bai_path = args.infile.with_suffix(".bam.bai")
        assert bai_path.is_file(), f"No BAM index file detected at path: {bai_path}"
        process_bam_file(args)
    elif args.mode == "plot":
        pass
    else:
        raise RuntimeError(f"Unsupported run mode set: {args.mode}")

    return 0


if __name__ == "__main__":
    main()
