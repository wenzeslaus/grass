/*!
 * \file lib/gis/cpu_cores.c
 *
 * \brief GIS Library - Number of physical CPU cores.
 *
 * SPDX-FileCopyrightText: 2026 Vaclav Petras
 * SPDX-FileCopyrightText: GRASS Development Team
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* sched_getaffinity, CPU_ISSET */
#endif
#include <sched.h>
#endif
#include <stdio.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#include <grass/gis.h>

static int logical_cpus(void)
{
#if defined(__linux__)
    cpu_set_t set;

    if (sched_getaffinity(0, sizeof(set), &set) == 0)
        return CPU_COUNT(&set);
#endif
#if defined(_WIN32)
    SYSTEM_INFO info;

    GetSystemInfo(&info);
    return info.dwNumberOfProcessors;
#elif defined(_SC_NPROCESSORS_ONLN)
    long n = sysconf(_SC_NPROCESSORS_ONLN);

    if (n > 0)
        return (int)n;
#endif
    return 1;
}

#if defined(__linux__)
/* Read a sysfs CPU list ("0,2-3") of the topology of a CPU into a set.
 * Returns 0 when the file is missing or holds no CPU number. */
static int read_cpu_list(int cpu, const char *name, cpu_set_t *cpus)
{
    char path[128];
    FILE *fp;
    int first;
    int found = 0;

    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s",
             cpu, name);
    CPU_ZERO(cpus);
    fp = fopen(path, "r");
    if (!fp)
        return 0;
    while (fscanf(fp, "%d", &first) == 1) {
        int last = first;
        int c = getc(fp);

        if (c == '-') {
            if (fscanf(fp, "%d", &last) != 1)
                break;
            c = getc(fp);
        }
        for (int i = first; i <= last; i++)
            if (i >= 0 && i < CPU_SETSIZE)
                CPU_SET(i, cpus);
        found = 1;
        if (c != ',')
            break;
    }
    fclose(fp);
    return found;
}

/* Count the cores among the CPUs this process may run on, so that a CPU
 * affinity mask limits the count like it limits the logical CPUs. A core
 * is the set of CPUs sharing it (its SMT siblings) and is counted at its
 * lowest CPU in the mask. The sets come from core_cpus_list, or from
 * thread_siblings_list on kernels before 5.3. */
static int linux_cores(void)
{
    cpu_set_t set, core;
    int n = 0;

    if (sched_getaffinity(0, sizeof(set), &set) != 0)
        return 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        int first;

        if (!CPU_ISSET(cpu, &set))
            continue;
        if (!read_cpu_list(cpu, "core_cpus_list", &core) &&
            !read_cpu_list(cpu, "thread_siblings_list", &core))
            return 0;
        CPU_AND(&core, &core, &set);
        for (first = 0; first < cpu && !CPU_ISSET(first, &core); first++)
            ;
        if (first == cpu)
            n++;
    }
    return n;
}
#endif

/*!
   \brief Get the number of physical CPU cores available to this process

   Logical CPUs of one core (SMT, Hyper-Threading) count once.

   On Linux, the count is limited to the CPUs in the affinity mask of the
   process. On macOS, it is the hw.physicalcpu sysctl value. Elsewhere,
   and where the topology is unknown, the number of logical CPUs is
   returned.

   \return number of physical cores (at least 1)

   \since version 8.6
 */
int G_available_cores(void)
{
    int n = 0;

#if defined(__linux__)
    n = linux_cores();
#elif defined(__APPLE__)
    int value;
    size_t size = sizeof(value);

    if (sysctlbyname("hw.physicalcpu", &value, &size, NULL, 0) == 0)
        n = value;
#endif
    if (n < 1)
        n = logical_cpus();
    return n < 1 ? 1 : n;
}
