/* Sudoku puzzle generator
 * Copyright (C) 2011 Daniel Beer <dlbeer@gmail.com>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <string.h>
#include <time.h>
#include <getopt.h>
#include <sys/time.h>

/* Compile this program with:
 *
 *    gcc -O3 -Wall -o sugen sugen.c
 *
 * The puzzle order corresponds to regular Sudoku by default (3), but
 * you can override it by specifying -DORDER=x at compile time. Orders
 * from 1 through 4 are supported.
 */

#ifndef ORDER
#define ORDER 3
#endif

#define DIM (ORDER * ORDER)
#define ELEMENTS (DIM * DIM)

/************************************************************************
 * Parsing and printing
 *
 * The parser is quite forgiving and designed so that it can also parse
 * grids produced by the pretty-printer.
 */

struct hline_def
{
    const char *start;
    const char *mid;
    const char *end;
    const char *major;
    const char *minor;
};

struct gridline_def
{
    struct hline_def top;
    struct hline_def major;
    struct hline_def minor;
    struct hline_def bottom;
    const char *vl_major;
    const char *vl_minor;
};

const static struct gridline_def ascii_def = {
    .top = {.start = "+", .mid = "-", .end = "+", .major = "+", .minor = "-"},
    .major = {.start = "+", .mid = "-", .end = "+", .major = "+", .minor = "-"},
    .minor = {.start = "|", .mid = ".", .end = "|", .major = "|", .minor = ":"},
    .bottom = {.start = "+", .mid = "-", .end = "+", .major = "+", .minor = "-"},
    .vl_major = "|",
    .vl_minor = ":"};

const static struct gridline_def utf8_def = {
    .top = {.start = "\xe2\x95\x94", .mid = "\xe2\x95\x90", .end = "\xe2\x95\x97", .major = "\xe2\x95\xa6", .minor = "\xe2\x95\xa4"},
    .major = {.start = "\xe2\x95\xa0", .mid = "\xe2\x95\x90", .end = "\xe2\x95\xa3", .major = "\xe2\x95\xac", .minor = "\xe2\x95\xaa"},
    .minor = {.start = "\xe2\x95\x9f", .mid = "\xe2\x94\x80", .end = "\xe2\x95\xa2", .major = "\xe2\x95\xab", .minor = "\xe2\x94\xbc"},
    .bottom = {.start = "\xe2\x95\x9a", .mid = "\xe2\x95\x90", .end = "\xe2\x95\x9d", .major = "\xe2\x95\xa9", .minor = "\xe2\x95\xa7"},
    .vl_major = "\xe2\x95\x91",
    .vl_minor = "\xe2\x94\x82"};

static int read_grid(uint8_t *grid)
{
    int x = 0;
    int y = 0;
    int c;
    int can_skip = 0;

    memset(grid, 0, sizeof(grid[0]) * ELEMENTS);

    while ((c = getchar()) >= 0)
    {
        if (c == '\n')
        {
            if (x > 0)
                y++;
            x = 0;
            can_skip = 0;
        }
        else if (c == '.' || c == '-')
        {
            can_skip = 0;
        }
        else if (c == '_' || c == '0')
        {
            x++;
        }
        else if (c == 0x82 || c == 0x91 || c == '|' || c == ':')
        {
            if (can_skip)
                x++;
            can_skip = 1;
        }
        else if (isalnum(c) && x < DIM && y < DIM)
        {
            int v;

            if (isdigit(c))
                v = c - '0';
            else if (isupper(c))
                v = c - 'A' + 10;
            else
                v = c - 'a' + 10;

            if (v >= 1 && v <= DIM)
            {
                grid[y * DIM + x] = v;
                x++;
                can_skip = 0;
            }
        }
    }

    if ((y <= DIM - 1) || ((y == DIM - 1) && (x < DIM)))
    {
        fprintf(stderr, "Too few cells in grid. Input ran out at "
                        "position (%d, %d)\n",
                x, y);
        return -1;
    }

    return 0;
}

static const char alphabet[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

static void print_grid(const uint8_t *grid)
{
    int y;

    for (y = 0; y < DIM; y++)
    {
        int x;

        for (x = 0; x < DIM; x++)
        {
            int v = grid[y * DIM + x];

            if (x)
                printf(""
                       " ");

            if (v)
                printf("%c", alphabet[v]);
            else
                printf("_");
        }

        printf("\n");
    }
}

static void draw_hline(const struct hline_def *def)
{
    int i;

    printf("%s", def->start);
    for (i = 0; i < DIM; i++)
    {
        printf("%s%s%s", def->mid, def->mid, def->mid);

        if (i + 1 < DIM)
            printf("%s", ((i + 1) % ORDER) ? def->minor : def->major);
    }
    printf("%s\n", def->end);
}

static void print_grid_pretty(const struct gridline_def *def,
                              const uint8_t *grid)
{
    int y;

    draw_hline(&def->top);

    for (y = 0; y < DIM; y++)
    {
        int x;

        for (x = 0; x < DIM; x++)
        {
            int v = grid[y * DIM + x];

            if (x % ORDER)
                printf("%s", def->vl_minor);
            else
                printf("%s", def->vl_major);

            if (v)
                printf(" %c ", alphabet[v]);
            else
                printf("   ");
        }

        printf("%s\n", def->vl_major);

        if (y + 1 < DIM)
        {
            if ((y + 1) % ORDER)
                draw_hline(&def->minor);
            else
                draw_hline(&def->major);
        }
    }

    draw_hline(&def->bottom);
}

/************************************************************************
 * Cell freedom analysis.
 *
 * Cell freedoms are sets, represented by bitfields. If bit N (counting
 * with LSB = 0) is set, then value (N+1) is present in the set.
 *
 * A cell freedom analysis results in a grid of sets, giving the immediately
 * allowable values in each cell position.
 *
 * If possible, it's cheaper to generate a freedom map for a new position by
 * modifying the previous position's freedom map, rather than rebuilding
 * from scratch.
 *
 * The search_least_free() function searches for the empty cell with the
 * smallest number of candidate values. It returns -1 if no empty cell was
 * found -- meaning that the grid is already solved.
 *
 * If the freedom for an empty cell is 0, this indicates that the grid is
 * unsolvable.
 */

typedef uint16_t set_t;

#define SINGLETON(v) (1 << ((v)-1))
#define ALL_VALUES ((1 << DIM) - 1)

static int count_bits(int x)
{
    int count = 0;

    while (x)
    {
        x &= x - 1;
        count++;
    }

    return count;
}

static void freedom_eliminate(set_t *freedom, int x, int y, int v)
{
    set_t mask = ~SINGLETON(v);
    int b;
    int i, j;
    set_t saved = freedom[y * DIM + x];

    b = x;
    for (i = 0; i < DIM; i++)
    {
        freedom[b] &= mask;
        b += DIM;
    }

    b = y * DIM;
    for (i = 0; i < DIM; i++)
        freedom[b + i] &= mask;

    b = (y - y % ORDER) * DIM + x - x % ORDER;
    for (i = 0; i < ORDER; i++)
    {
        for (j = 0; j < ORDER; j++)
            freedom[b + j] &= mask;

        b += DIM;
    }

    freedom[y * DIM + x] = saved;
}

static void init_freedom(const uint8_t *problem, set_t *freedom)
{
    int x, y;

    for (x = 0; x < ELEMENTS; x++)
        freedom[x] = ALL_VALUES;

    for (y = 0; y < DIM; y++)
        for (x = 0; x < DIM; x++)
        {
            int v = problem[y * DIM + x];

            if (v)
                freedom_eliminate(freedom, x, y, v);
        }
}

static int sanity_check(const uint8_t *problem, const set_t *freedom)
{
    int i;

    for (i = 0; i < ELEMENTS; i++)
    {
        int v = problem[i];

        if (v)
        {
            set_t f = freedom[i];

            if (!(f & SINGLETON(v)))
                return -1;
        }
    }

    return 0;
}

static int search_least_free(const uint8_t *problem, const set_t *freedom)
{
    int i;
    int best_index = -1;
    int best_score = -1;

    for (i = 0; i < ELEMENTS; i++)
    {
        int v = problem[i];

        if (!v)
        {
            int score = count_bits(freedom[i]);

            if (best_score < 0 || score < best_score)
            {
                best_index = i;
                best_score = score;
            }
        }
    }

    return best_index;
}

/************************************************************************
 * Set-oriented freedom analysis.
 *
 * In normal freedom analysis, we find candidate values for each cell. In
 * set-oriented freedom analysis, we find candidate cells for each value.
 * There are 3 * DIM sets to consider (DIM boxes, rows and columns).
 *
 * The sofa() function returns the size of the smallest set of positions
 * found, along with a list of those positions and a value which can occupy
 * all of those positions. It returns -1 if a set of positions couldn't
 * be found.
 *
 * If it returns 0, this indicates that there exists a set, missing a value,
 * with no possible positions for that value -- the grid is therefore
 * unsolvable.
 *
 * If the program is compiled with -DNO_SOFA, this analysis is not used.
 */

#ifndef NO_SOFA
struct sofa_context
{
    const uint8_t *grid;
    const set_t *freedom;

    int best[DIM];
    int best_size;
    int best_value;
};

static void sofa_set(struct sofa_context *ctx, const int *set)
{
    int count[DIM];
    int i;
    int best = -1;
    set_t missing = ALL_VALUES;

    /* Find out what's missing from the set, and how many available
	 * slots for each missing number.
	 */
    memset(count, 0, sizeof(count));
    for (i = 0; i < DIM; i++)
    {
        int v = ctx->grid[set[i]];

        if (v)
        {
            missing &= ~SINGLETON(v);
        }
        else
        {
            set_t freedom = ctx->freedom[set[i]];
            int j;

            for (j = 0; j < DIM; j++)
                if (freedom & (1 << j))
                    count[j]++;
        }
    }

    /* Look for the missing number with the fewest available slots. */
    for (i = 0; i < DIM; i++)
        if ((missing & (1 << i)) &&
            (best < 0 || count[i] < count[best]))
            best = i;

    /* Did we find anything? */
    if (best < 0)
        return;

    /* If it's better than anything we've found so far, save the result */
    if (ctx->best_size < 0 || count[best] < ctx->best_size)
    {
        int j = 0;
        set_t mask = 1 << best;

        ctx->best_value = best + 1;
        ctx->best_size = count[best];

        for (i = 0; i < DIM; i++)
            if (!ctx->grid[set[i]] &&
                (ctx->freedom[set[i]] & mask))
                ctx->best[j++] = set[i];
    }
}

static int sofa(const uint8_t *grid, const set_t *freedom, int *set, int *value)
{
    struct sofa_context ctx;
    int i;

    memset(&ctx, 0, sizeof(ctx));
    ctx.grid = grid;
    ctx.freedom = freedom;
    ctx.best_size = -1;
    ctx.best_value = -1;

    for (i = 0; i < DIM; i++)
    {
        int b = (i / ORDER) * ORDER * DIM + (i % ORDER) * ORDER;
        int set[DIM];
        int j;

        for (j = 0; j < DIM; j++)
            set[j] = j * DIM + i;
        sofa_set(&ctx, set);

        for (j = 0; j < DIM; j++)
            set[j] = i * DIM + j;
        sofa_set(&ctx, set);

        for (j = 0; j < DIM; j++)
            set[j] = b + (j / ORDER) * DIM + j % ORDER;
        sofa_set(&ctx, set);
    }

    memcpy(set, ctx.best, sizeof(ctx.best));
    *value = ctx.best_value;
    return ctx.best_size;
}
#endif

/************************************************************************
 * Solver
 *
 * The solver works using recursive backtracking. The general idea is to
 * find the cell with the smallest possible number of candidate values, and
 * to try each candidate, recursively solving until we find a solution.
 *
 * However, in cases where a cell has multiple candidates, we also consider
 * set-oriented backtracking -- choosing a value and trying each candidate
 * position. If this yields a smaller branching factor (it often eliminates
 * the need for backtracking), we try it instead.
 *
 * We keep searching until we've either found two solutions (demonstrating
 * that the grid does not have a unique solution), or we exhaust the search
 * tree.
 *
 * We also calculate a branch-difficulty score:
 *
 *    Sigma [(B_i - 1) ** 2]
 *
 * Where B_i are the branching factors at each node in the search tree
 * following the path from the root to the solution. A puzzle that could
 * be solved without backtracking has a branch-difficulty of 0.
 *
 * The final difficulty is:
 *
 *    Difficulty = B * C + E
 *
 * Where B is the branch-difficulty, E is the number of empty cells, and C
 * is the first power of ten greater than the number of elements.
 */

struct solve_context
{
    uint8_t problem[ELEMENTS];
    int count;
    uint8_t *solution;
    int branch_score;
};

static void solve_recurse(struct solve_context *ctx, const set_t *freedom,
                          int diff)
{
    set_t new_free[ELEMENTS];
    set_t mask;
    int r;
    int i;
    int bf;

    r = search_least_free(ctx->problem, freedom);
    if (r < 0)
    {
        if (!ctx->count)
        {
            ctx->branch_score = diff;
            if (ctx->solution)
                memcpy(ctx->solution, ctx->problem,
                       ELEMENTS * sizeof(ctx->solution[0]));
        }

        ctx->count++;
        return;
    }

    mask = freedom[r];

#ifndef NO_SOFA
    /* If we can't determine a cell value, see if set-oriented
	 * backtracking provides a smaller branching factor.
	 */
    if (mask & (mask - 1))
    {
        int set[DIM];
        int value;
        int size;

        size = sofa(ctx->problem, freedom, set, &value);
        if (size >= 0 && size < count_bits(mask))
        {
            bf = size - 1;
            diff += bf * bf;

            for (i = 0; i < size; i++)
            {
                int s = set[i];

                memcpy(new_free, freedom, sizeof(new_free));
                freedom_eliminate(new_free,
                                  s % DIM, s / DIM, value);

                ctx->problem[s] = value;
                solve_recurse(ctx, new_free, diff);
                ctx->problem[s] = 0;

                if (ctx->count >= 2)
                    return;
            }

            return;
        }
    }
#endif

    /* Otherwise, fall back to cell-oriented backtracking. */
    bf = count_bits(mask) - 1;
    diff += bf * bf;

    for (i = 0; i < DIM; i++)
        if (mask & (1 << i))
        {
            memcpy(new_free, freedom, sizeof(new_free));
            freedom_eliminate(new_free, r % DIM, r / DIM, i + 1);
            ctx->problem[r] = i + 1;
            solve_recurse(ctx, new_free, diff);

            if (ctx->count >= 2)
                return;
        }

    ctx->problem[r] = 0;
}

static int solve(const uint8_t *problem, uint8_t *solution, int *diff)
{
    struct solve_context ctx;
    set_t freedom[ELEMENTS];

    memcpy(ctx.problem, problem, sizeof(ctx.problem));
    ctx.count = 0;
    ctx.branch_score = 0;
    ctx.solution = solution;

    init_freedom(problem, freedom);
    if (sanity_check(problem, freedom) < 0)
        return -1;

    solve_recurse(&ctx, freedom, 0);

    /* Calculate a difficulty score */
    if (diff)
    {
        int empty = 0;
        int mult = 1;
        int i;

        for (i = 0; i < ELEMENTS; i++)
            if (!problem[i])
                empty++;

        while (mult <= ELEMENTS)
            mult *= 10;

        *diff = ctx.branch_score * mult + empty;
    }

    return ctx.count - 1;
}

/************************************************************************
 * Grid generator
 *
 * We generate grids using a backtracking algorithm similar to the basic
 * solver algorithm. At each step, choose a cell with the smallest number
 * of possible values, and try each value, solving recursively. The key
 * difference is that the values are tested in a random order.
 *
 * An empty grid can be initially populated with a large number of values
 * without backtracking. In the ORDER == 3 case, we can easily fill the
 * whole top band the the first column before resorting to backtracking.
 */

static int pick_value(set_t set)
{
    int x = random() % count_bits(set);
    int i;

    for (i = 0; i < DIM; i++)
        if (set & (1 << i))
        {
            if (!x)
                return i + 1;
            x--;
        }

    return 0;
}

static void choose_b1(uint8_t *problem)
{
    set_t set = ALL_VALUES;
    int i, j;

    for (i = 0; i < ORDER; i++)
        for (j = 0; j < ORDER; j++)
        {
            int v = pick_value(set);

            problem[i * DIM + j] = v;
            set &= ~SINGLETON(v);
        }
}

#if ORDER == 3
static void choose_b2(uint8_t *problem)
{
    set_t used[ORDER];
    set_t chosen[ORDER];
    set_t set_x, set_y;
    int i, j;

    memset(used, 0, sizeof(used));
    memset(chosen, 0, sizeof(chosen));

    /* Gather used values from B1 by box-row */
    for (i = 0; i < ORDER; i++)
        for (j = 0; j < ORDER; j++)
            used[i] |= SINGLETON(problem[i * DIM + j]);

    /* Choose the top box-row for B2 */
    set_x = used[1] | used[2];
    for (i = 0; i < ORDER; i++)
    {
        int v = pick_value(set_x);
        set_t mask = SINGLETON(v);

        chosen[0] |= mask;
        set_x &= ~mask;
    }

    /* Choose values for the middle box-row, as long as we can */
    set_x = (used[0] | used[2]) & ~chosen[0];
    set_y = (used[0] | used[1]) & ~chosen[0];

    while (count_bits(set_y) > 3)
    {
        int v = pick_value(set_x);
        set_t mask = SINGLETON(v);

        chosen[1] |= mask;
        set_x &= ~mask;
        set_y &= ~mask;
    }

    /* We have no choice for the remainder */
    chosen[1] |= set_x & ~set_y;
    chosen[2] |= set_y;

    /* Permute the triplets in each box-row */
    for (i = 0; i < ORDER; i++)
    {
        set_t set = chosen[i];
        int j;

        for (j = 0; j < ORDER; j++)
        {
            int v = pick_value(set);

            problem[i * DIM + j + ORDER] = v;
            set &= ~SINGLETON(v);
        }
    }
}

static void choose_b3(uint8_t *problem)
{
    int i;

    for (i = 0; i < ORDER; i++)
    {
        set_t set = ALL_VALUES;
        int j;

        /* Eliminate already-used values in this row */
        for (j = 0; j + ORDER < DIM; j++)
            set &= ~SINGLETON(problem[i * DIM + j]);

        /* Permute the remaining values in the last box-row */
        for (j = 0; j < ORDER; j++)
        {
            int v = pick_value(set);

            problem[i * DIM + DIM - ORDER + j] = v;
            set &= ~SINGLETON(v);
        }
    }
}
#endif /* ORDER == 3 */

static void choose_col1(uint8_t *problem)
{
    set_t set = ALL_VALUES;
    int i;

    for (i = 0; i < ORDER; i++)
        set &= ~SINGLETON(problem[i * DIM]);

    for (; i < DIM; i++)
    {
        int v = pick_value(set);

        problem[i * DIM] = v;
        set &= ~SINGLETON(v);
    }
}

static int choose_rest(uint8_t *grid, const set_t *freedom)
{
    int i = search_least_free(grid, freedom);
    set_t set;

    if (i < 0)
        return 0;

    set = freedom[i];
    while (set)
    {
        set_t new_free[ELEMENTS];
        int v = pick_value(set);

        set &= ~SINGLETON(v);
        grid[i] = v;

        memcpy(new_free, freedom, sizeof(new_free));
        freedom_eliminate(new_free, i % DIM, i / DIM, v);

        if (!choose_rest(grid, new_free))
            return 0;
    }

    grid[i] = 0;
    return -1;
}

static void choose_grid(uint8_t *grid)
{
    set_t freedom[ELEMENTS];

    memset(grid, 0, sizeof(grid[0]) * ELEMENTS);

    choose_b1(grid);
#if ORDER == 3
    choose_b2(grid);
    choose_b3(grid);
#endif
    choose_col1(grid);

    init_freedom(grid, freedom);
    choose_rest(grid, freedom);
}

/************************************************************************
 * Puzzle generator
 *
 * To generate a puzzle, we start with a solution grid, and an initial
 * puzzle (which may be the same as the solution). We try altering the
 * puzzle by either randomly adding a pair of clues from the solution, or
 * randomly removing a pair of clues. After each alteration, we check to
 * see if we have a valid puzzle. If it is, and it's more difficult than
 * anything we've encountered so far, save it as the best puzzle.
 *
 * To avoid getting stuck in local minima in the space of puzzles, we allow
 * the algorithm to wander for a few steps before starting again from the
 * best-so-far puzzle.
 */

static int harden_puzzle(const uint8_t *solution, uint8_t *puzzle,
                         int max_iter, int max_score, int target_score)
{
    int best = 0;
    int i;

    solve(puzzle, NULL, &best);

    for (i = 0; i < max_iter; i++)
    {
        uint8_t next[ELEMENTS];
        int j;

        memcpy(next, puzzle, sizeof(next));

        for (j = 0; j < DIM * 2; j++)
        {
            int c = random() % ELEMENTS;
            int s;

            if (random() & 1)
            {
                next[c] = solution[c];
                next[ELEMENTS - c - 1] =
                    solution[ELEMENTS - c - 1];
            }
            else
            {
                next[c] = 0;
                next[ELEMENTS - c - 1] = 0;
            }

            if (!solve(next, NULL, &s) &&
                s > best && (s <= max_score || max_score < 0))
            {
                memcpy(puzzle, next,
                       sizeof(puzzle[0]) * ELEMENTS);
                best = s;

                if (target_score >= 0 && s >= target_score)
                    return best;
            }
        }
    }

    return best;
}

/************************************************************************
 * Command-line user interface
 */

struct options
{
    int max_iter;
    int target_diff;
    int max_diff;
    const struct gridline_def *gl_def;
    const char *action;
};

static void usage(const char *progname)
{
    printf("usage: %s [options] <action>\n\n"
           "Options may be any of the following:\n"
           "    -i <iterations>    Maximum iterations for puzzle generation.\n"
           "    -m <score>         Maximum difficulty for puzzle generation.\n"
           "    -t <score>         Target difficulty for puzzle generation.\n"
           "    -u                 Use UTF-8 line-drawing characters.\n"
           "    -a                 Use ASCII line-drawing characters.\n"
           "\n"
           "Action should be one of:\n"
           "    solve              Read a grid from stdin and solve it.\n"
           "    examine            Read a grid and estimate the difficulty.\n"
           "    print              Read a grid and reformat it.\n"
           "    generate           Generate and print a new puzzle.\n"
           "    harden             Read an existing puzzle and make it harder.\n"
           "    gen-grid           Generate a valid grid.\n",
           progname);
}

static void version(void)
{
    printf(
        "Sudoku puzzle generator, 10 Jun 2011\n"
        "Copyright (C) 2011 Daniel Beer <dlbeer@gmail.com>\n"
        "\n"
        "Permission to use, copy, modify, and/or distribute this software for any\n"
        "purpose with or without fee is hereby granted, provided that the above\n"
        "copyright notice and this permission notice appear in all copies.\n"
        "\n"
        "THE SOFTWARE IS PROVIDED \"AS IS\" AND THE AUTHOR DISCLAIMS ALL WARRANTIES\n"
        "WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF\n"
        "MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR\n"
        "ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES\n"
        "WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN\n"
        "ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF\n"
        "OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.\n");
}

static int parse_options(int argc, char **argv, struct options *o)
{
    const static struct option longopts[] = {
        {"help", 0, 0, 'H'},
        {"version", 0, 0, 'V'},
        {NULL, 0, 0, 0}};
    int i;

    memset(o, 0, sizeof(*o));
#if ORDER <= 3
    o->max_iter = 200;
#else
    o->max_iter = 20;
#endif
    o->target_diff = -1;
    o->max_diff = -1;

    while ((i = getopt_long(argc, argv, "i:t:m:ua", longopts, NULL)) >= 0)
        switch (i)
        {
        case 'i':
            o->max_iter = atoi(optarg);
            break;

        case 't':
            o->target_diff = atoi(optarg);
            break;

        case 'm':
            o->max_diff = atoi(optarg);
            break;

        case 'u':
            o->gl_def = &utf8_def;
            break;

        case 'a':
            o->gl_def = &ascii_def;
            break;

        case 'H':
            usage(argv[0]);
            exit(0);
            break;

        case 'V':
            version();
            exit(0);

        case '?':
            fprintf(stderr, "Try --help for usage "
                            "information.\n");
            return -1;
        }

    argc -= optind;
    argv += optind;

    if (argc < 1)
    {
        version();
        printf("\n");
        fprintf(stderr, "You need to specify an action. "
                        "Try --help.\n");
        return -1;
    }

    o->action = argv[0];
    return 0;
}

static void print_grid_opt(const struct options *o, const uint8_t *grid)
{
    if (o->gl_def)
        print_grid_pretty(o->gl_def, grid);
    else
        print_grid(grid);
}

static int action_se(const struct options *o)
{
    uint8_t grid[ELEMENTS];
    uint8_t solution[ELEMENTS];
    int diff;
    int r;

    if (read_grid(grid) < 0)
        return -1;

    r = solve(grid, solution, &diff);
    if (r < 0)
    {
        printf("Grid is unsolvable\n");
        return -1;
    }

    if (*o->action == 's' || *o->action == 'S')
    {
        print_grid_opt(o, solution);
        printf("\n");
    }

    if (r > 0)
    {
        printf("Solution is not unique\n");
        return -1;
    }

    printf("Unique solution. Difficulty: %d\n", diff);
    return 0;
}

static int action_print(const struct options *o)
{
    uint8_t grid[ELEMENTS];

    if (read_grid(grid) < 0)
        return -1;

    print_grid_opt(o, grid);
    return 0;
}

static int action_gen_grid(const struct options *o)
{
    uint8_t grid[ELEMENTS];

    choose_grid(grid);
    print_grid_opt(o, grid);
    return 0;
}

static int action_harden(const struct options *o)
{
    uint8_t solution[ELEMENTS];
    uint8_t grid[ELEMENTS];
    int r;
    int old_diff;
    int new_diff;

    if (read_grid(grid) < 0)
        return -1;

    r = solve(grid, solution, &old_diff);
    if (r < 0)
    {
        printf("Grid is unsolvable\n");
        return -1;
    }

    if (r)
        memcpy(grid, solution, sizeof(grid[0]) * ELEMENTS);

    new_diff = harden_puzzle(solution, grid, o->max_iter,
                             o->max_diff, o->target_diff);

    print_grid_opt(o, grid);
    printf("\nDifficulty: %d\n", new_diff);

    if (r)
        printf("Original puzzle was not uniquely solvable\n");
    else
        printf("Original difficulty: %d\n", old_diff);

    return 0;
}

static int action_generate(const struct options *o)
{
    uint8_t puzzle[ELEMENTS];
    uint8_t grid[ELEMENTS];
    int diff;

    choose_grid(grid);
    memcpy(puzzle, grid, ELEMENTS * sizeof(puzzle[0]));

    diff = harden_puzzle(grid, puzzle, o->max_iter,
                         o->max_diff, o->target_diff);
    print_grid_opt(o, puzzle);
    printf("\nDifficulty: %d\n", diff);
    return 0;
}

struct action
{
    const char *name;
    int (*func)(const struct options *o);
};

static const struct action actions[] = {
    {"solve", action_se},
    {"examine", action_se},
    {"print", action_print},
    {"gen-grid", action_gen_grid},
    {"harden", action_harden},
    {"generate", action_generate},
    {NULL, NULL}};

int main(int argc, char **argv)
{
    struct timeval time;
    gettimeofday(&time, NULL);

    struct options o;
    const struct action *a = actions;

    if (parse_options(argc, argv, &o) < 0)
        return -1;

    while (a->name && strcasecmp(o.action, a->name))
        a++;

    if (!a->name)
    {
        fprintf(stderr, "Unknown action: %s. Try --help.\n", o.action);
        return -1;
    }

    srandom((time.tv_sec * 1000) + (time.tv_usec / 1000));
    return a->func(&o);
}
