#!/usr/bin/env python3
import argparse
import contextlib as ctl
import pathlib as pl
import re
import sys
import time

import numpy as np
import pandas as pd
import pysam

#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#from matplotlib.collections import PatchCollection
#import seaborn as sns

#matplotlib.use("agg")

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
        "--repeatmasker",
        "-rm",
        type=lambda x: pl.Path(x).resolve(strict=True),
        default=None,
        dest="repeatmasker",
        help="Path to repeatmasker output to add to the plot. Default: None"
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
        "--out-plot",
        "-op",
        type=lambda x: pl.Path(x).resolve(strict=False),
        dest="out_plot",
        help="Path to plot output file.",
        default=None
    )
    io_group.add_argument(
        "--out-hdf-cache",
        "-hdf",
        type=lambda x: pl.Path(x).resolve(strict=False),
        dest="out_hdf_cache",
        help="Path to output cache file (HDF format), required for plotting.",
        default=pl.Path(".").resolve(strict=False).joinpath("nucfreqtwo.cache.h5"),
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
        dest="min_discordant_bases",
        help="min number of discordant bases to report in output BED. Default: 2"
    )

    runtime_group.add_argument(
        "--flag-discordant-pct",
        "-fdp",
        type=int,
        default=10,
        dest="discordant_pct",
        help="Flag criteria: minimum percent of discordant bases. Default: 10 %"
    )

    runtime_group.add_argument(
        "--flag-discordant-abs",
        "-fdc",
        type=int,
        default=2,
        dest="discordant_abs",
        help="Flag criteria: minimum absolute count for second most common base. Default: 2"
    )

    runtime_group.add_argument(
        "--flag-min-interval",
        "-fmi",
        type=int,
        default=500,
        dest="min_interval",
        help="Flag criteria: minimum interval size. Default: 500 bp"
    )

    runtime_group.add_argument(
        "--flag-store-window",
        "-fsw",
        type=int,
        default=50000,
        dest="store_window",
        help="Store this window around flagged regions. Default: 50 kbp"
    )

    runtime_group.add_argument(
        "--flag-num-hets",
        "-fnh",
        type=int,
        default=5,
        dest="num_hets",
        help="Flag criteria: minimum number of HETs per interval. Default: 5"
    )

    # === new arguments ===
    runtime_group.add_argument(
        "--min-region-size",
        "-mrs",
        type=int,
        default=0,
        dest="min_region_size",
        help=(
            "If no regions are specified by the user and all sequences in the BAM "
            "input file are considered, skip those with size smaller than this value. "
            "Default: 0 (bp)"
        )
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


def __isolated_constants():

    raise NotImplementedError

    # not used - purpose unclear
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

    return None


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
        regions.append((chrom, int(start), int(end), -1, -1))
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
                regions.append((chrom, start, end, -1, -1))
    return None


def get_regions_from_bam(bam_file, threads, min_region_size, all_positions, refs, regions):

    with pysam.AlignmentFile(bam_file, threads=threads) as bam:
        for ref_name, ref_length in zip(bam.references, bam.lengths):
            if ref_length > int(5e6):
                continue
            if ref_length < min_region_size:
                continue
            # logically, I think this should not happen
            # because we only end up here if the user did
            # not specify regions; mimicking the original
            # control flow for now ...
            if ref_name in refs:
                continue
            if all_positions:
                regions.append((ref_name, 0, ref_length, -1))
                continue
            else:
                min_start = ref_length
                max_end = 0
                read_count = 0
                for read_count, read in enumerate(bam.fetch(contig=ref_name), start=0):
                    min_start = min(min_start, read.reference_start)
                    max_end = max(max_end, read.reference_end)
                # if the contig is not covered at all, need to switch
                # start and end
                if min_start > max_end:
                    assert read_count == 0, f"{ref_name} / {ref_length} / {read_count}: {min_start} - {max_end}"
                    min_start, max_end = max_end, min_start
                assert min_start < max_end, f"{ref_name} / {ref_length}: {min_start} - {max_end}"
                regions.append((ref_name, min_start, max_end, ref_length, read_count))

            ### === BEGIN: CHANGED CONTROL FLOW
            # this forced iteration is a bit annoying /
            # unnecessary if "all_positions" is set.
            # Only kept in code because of the call
            # to "getSoft" that is of unclear importance

            #for read in bam.fetch(contig=ref_name):
                # this uses the default of 0 for the
                # groups parameter, whereas later, the
                # soft/hard-masking is computed per "group"
                # (i.e., reference sequence?); smells like
                # unintended behavior ...

            #    soft += getSoft(read)
            ### === END: CHANGED CONTROL FLOW

    return None


def check_regions_to_flag(het_positions, flag_ivsize, flag_numhet):

    het_positions["start"] = het_positions.index.values
    het_positions["end"] = het_positions["start"] + 1
    het_positions["iv_end"] = het_positions["start"] + flag_ivsize
    # need the following indicator to track
    # how many rows were merged per interval
    het_positions["num_hets"] = 1
    het_positions["iv_idx"] = (het_positions["start"] > het_positions["iv_end"].shift().cummax()).cumsum()
    flag_regions = het_positions.groupby("iv_idx").agg(
        {
            "start":"min", "end": "max",
            "num_hets": "sum",
            "het_pct": "median"
        }
    )
    flag_regions = flag_regions.loc[flag_regions["num_hets"] >= flag_numhet, :].copy()
    return flag_regions


def compute_nucleotide_frequencies(
        bam_file, threads, regions, track_soft_masked, soft_min_clip,
        flag_ratio, flag_absolute, flag_ivsize, flag_numhet, flag_store_window,
        cache_file):

    GROUPS = 0

    cache_file.parent.mkdir(exist_ok=True, parents=True)
    contig_group_table = []

    half_window = flag_store_window // 2

    BYTE_TO_GB = 1024 ** 3

    with ctl.ExitStack() as exs:

        bam = exs.enter_context(pysam.AlignmentFile(bam_file, threads=threads))
        hdf = exs.enter_context(pd.HDFStore(cache_file, mode="w", complevel=9, complib="blosc"))

        for contig, start, end, ref_length, read_count in sorted(regions):
            proc_start = time.perf_counter()
            # coverage = tuple of 4
            # arrays for A C G T
            try:
                coverage = bam.count_coverage(
                    contig, start=start, stop=end,
                    read_callback="nofilter", quality_threshold=None
                )
            except OverflowError:
                sys.stderr.write(f"\nOverflowError while processing: {contig} - {start} - {end}\n")
                raise
            except ValueError:
                sys.stderr.write(f"\nValueError while processing: {contig} - {start} - {end}\n")
                raise
            assert len(coverage) == 4
            contiglen = len(coverage[0])

            # unclear how this could be zero?
            if contiglen > 0:
                # by default, the numeric index of the dataframe
                # indexes the positions in the sequence
                nucfreq = pd.DataFrame(
                    np.zeros((contiglen, 4), dtype=np.int32),
                    columns=["A", "C", "G", "T"],
                    index=np.arange(start, end, 1, dtype=np.int32)
                )
                nucfreq["A"] = coverage[0]
                nucfreq["C"] = coverage[1]
                nucfreq["G"] = coverage[2]
                nucfreq["T"] = coverage[3]

                sort = np.flip(np.sort(nucfreq[["A", "C", "G", "T"]].values), 1)
                # consider dropping third and fourth, not used at all ...
                nucfreq["first"] = sort[:, 0]
                nucfreq["second"] = sort[:, 1]
                nucfreq["third"] = sort[:, 2]
                nucfreq["fourth"] = sort[:, 3]
                nucfreq["het_pct"] = ((nucfreq["second"] / (nucfreq["first"] + nucfreq["second"])) * 100).round(1)
                nucfreq["het_pct"].fillna(0., inplace=True)

                mem_gb = (nucfreq.memory_usage(index=True).sum() / BYTE_TO_GB).round(2)

                discordant_pct = nucfreq["het_pct"] >= flag_ratio
                discordant_abs = nucfreq["second"] >= flag_absolute

                # this should usually be small-ish ...
                discordant = nucfreq.loc[discordant_pct & discordant_abs, :].copy()

                discordant_pos = discordant.shape[0]
                num_flagged_regions = 0
                num_flagged_bp = 0
                if not discordant.empty:
                    flag_regions = check_regions_to_flag(discordant.copy(), flag_ivsize, flag_numhet)
                    discordant["flagged"] = 0
                    discordant["region_id"] = 0
                    region_id = 1
                    for row in flag_regions.itertuples():
                        select_start = discordant.index >= row.start
                        select_end = discordant.index < row.end
                        discordant.loc[select_start & select_end, "flagged"] = 1
                        discordant.loc[select_start & select_end, "region_id"] = region_id
                        num_flagged_bp += (row.end - row.start)

                        store_start = nucfreq.index > row.start - half_window
                        store_end = nucfreq.index < row.end + half_window

                        hdf.put(
                            f"group_{GROUPS}/region_window/region_{region_id}",
                            nucfreq.loc[store_start & store_end, :],
                            format="fixed"
                        )

                        num_flagged_regions += 1
                        region_id += 1

                    hdf.put(
                        f"group_{GROUPS}/positions",
                        discordant,
                        format="fixed"
                    )
                    if num_flagged_regions > 0:
                        hdf.put(
                            f"group_{GROUPS}/regions",
                            flag_regions,
                            format="fixed"
                        )

                if track_soft_masked:
                    masked_positions = []
                    for read in bam.fetch(contig, start, end):
                        masked_positions += getSoft(read, group=GROUPS)
                    masked_positions = pd.DataFrame(
                            masked_positions,
                            columns=["contig", "side", "value", "position", "read", "group"]
                        )
                    masked_positions = masked_positions[masked_positions.value >= soft_min_clip].copy()
                    if not masked_positions.empty:
                        masked_positions.sort_values(by=["contig", "position"], inplace=True)
                        hdf.put(
                            f"group_{GROUPS}/masked",
                            masked_positions,
                            format="fixed"
                        )

                proc_end = time.perf_counter()
                proc_time = proc_end - proc_start

                contig_group_table.append(
                    (
                        contig, start, end, ref_length, GROUPS, read_count,
                        discordant_pos, num_flagged_regions, num_flagged_bp,
                        proc_time, mem_gb
                    )
                )

                # the GROUPS value is used later for plotting.
                # It makes thus sense to only increment it if
                # the respective contig indeed had any data
                GROUPS += 1

            else:
                contig_group_table.append(
                    (
                        contig, start, end, ref_length, -1, -1, 0, 0, 0, 0, 0
                    )
                )

        group_table = pd.DataFrame.from_records(
            contig_group_table,
            columns=[
                "contig", "start", "end", "length", "group", "read_count",
                "num_discordant_pos", "num_flagged_regions", "flagged_bp",
                "proc_time_sec", "mem_use_gb"
            ]
        )
        hdf.put("group_table", group_table, format="fixed")

    return


def process_bam_file(args):

    refs = {}
    regions = []
    if args.regions is not None:
        parse_user_regions(args.regions, refs, regions)
    if args.in_bed_regions is not None:
        parse_regions_from_bed_input(args.in_bed_regions, refs, regions)

    if not regions:
        # implies that all regions from the BAM file are to be processed
        get_regions_from_bam(
            args.infile, args.threads, args.min_region_size,
            args.all_positions, refs, regions
        )

    compute_nucleotide_frequencies(
        args.infile, args.threads, regions,
        args.soft, args.minclip,
        args.discordant_pct, args.discordant_abs,
        args.min_interval, args.num_hets, args.store_window,
        args.out_hdf_cache
    )

    return None


def plot_nucleotide_frequencies():

    raise NotImplementedError


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

    return None


def main():

    args = parse_command_line()

    if args.mode == "process":
        # check that BAM index exists at standard path
        bai_path = args.infile.with_suffix(".bam.bai")
        assert bai_path.is_file(), f"No BAM index file detected at path: {bai_path}"
        process_bam_file(args)
    elif args.mode == "plot":
        plot_nucleotide_frequencies()
    else:
        raise RuntimeError(f"Unsupported run mode set: {args.mode}")

    return 0


if __name__ == "__main__":
    main()
