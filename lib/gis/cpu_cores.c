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
static int read_int(int cpu, const char *name)
{
    char path[128];
    FILE *fp;
    int value = -1;

    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s",
             cpu, name);
    fp = fopen(path, "r");
    if (!fp)
        return -1;
    if (fscanf(fp, "%d", &value) != 1)
        value = -1;
    fclose(fp);
    return value;
}

/* Count distinct (package, core) pairs among the CPUs this process may
 * run on, so that a CPU affinity mask limits the count like it limits
 * the logical CPUs. */
static int linux_cores(void)
{
    cpu_set_t set;
    int keys[2 * CPU_SETSIZE];
    int n = 0;

    if (sched_getaffinity(0, sizeof(set), &set) != 0)
        return 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        int package, core, i;

        if (!CPU_ISSET(cpu, &set))
            continue;
        package = read_int(cpu, "physical_package_id");
        core = read_int(cpu, "core_id");
        if (package < 0 || core < 0) {
            n = 0;
            break;
        }
        for (i = 0; i < n; i++)
            if (keys[2 * i] == package && keys[2 * i + 1] == core)
                break;
        if (i == n) {
            keys[2 * n] = package;
            keys[2 * n + 1] = core;
            n++;
        }
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
