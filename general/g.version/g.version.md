## DESCRIPTION

*g.version* prints to standard output the GRASS version number, date,
the GRASS copyright (**-c** flag), and GRASS build information
(**-b** flag).

## NOTES

This program requires no command line arguments; the user simply types
*g.version* on the command line to see the version number and date of
the GRASS software currently being run by the user.

Information about GRASS core [GIS
Library](https://grass.osgeo.org/programming8/gislib.html) can be
printed by **-r** flag.

Version numbers of additional libraries like [PROJ](https://proj.org/),
[GDAL/OGR](https://gdal.org/) or [GEOS](https://trac.osgeo.org/geos) are
printed by **-e** flag. For PROJ, the flag also reports the version of
the library and the path of the database (`proj.db`) in use at runtime
(`proj_runtime`, `proj_db`). The library can differ from the compile-time
version, and the database can come from another PROJ installation, for
example, when another package sets `PROJ_DATA`. Such a mismatch is a
common cause of errors mentioning `proj.db`. An empty `proj_db` (`null`
in JSON) means PROJ could not open any database. The plain output then
also says whether a `proj.db` was found but rejected or not found at all.

See also function `version()` from [Python Scripting
Library](https://grasswiki.osgeo.org/wiki/GRASS_Python_Scripting_Library).

```python
import grass.script as gs

print(gs.version())
```

## EXAMPLES

### Basic info

```sh
g.version

GRASS 8.4.0 (2024)
```

### GIS Library info

```sh
g.version -r

GRASS 8.4.0 (2024)
libgis revision: c9e8576cf
libgis date: 2024-04-27T09:38:49+00:00
```

### Full info

<!-- markdownlint-disable MD046 -->
=== "Command line"

    ```sh
    g.version -re format=shell
    ```

    Possible output:

    ```text
    version=8.4.0
    date=2024
    revision=d57f40906
    build_date=2024-05-23
    build_platform=x86_64-pc-linux-gnu
    build_off_t_size=8
    libgis_revision=c9e8576cf
    libgis_date=2024-04-27T09:38:49+00:00
    proj=9.4.0
    proj_runtime=9.4.0
    proj_db=/usr/share/proj/proj.db
    gdal=3.4.3
    geos=3.9.2
    sqlite=3.36.0
    ```

=== "Python (grass.script)"

    ```python
    import grass.script as gs

    # Run g.version with JSON output and the -re flags
    data = gs.parse_command(
        "g.version",
        flags="re",
        format="json",
    )

    print(data["version"])
    ```

    Possible output:

    ```text
    8.4.0
    ```

    The whole JSON may look like this:

    ```json
    {
        "version": "8.4.0",
        "date": "2024",
        "revision": "d57f40906",
        "build_date": "2024-05-23",
        "build_platform": "x86_64-pc-linux-gnu",
        "build_off_t_size": 8,
        "libgis_revision": "c9e8576cf",
        "libgis_date": "2024-04-27T09:38:49+00:00",
        "proj": "9.4.0",
        "proj_runtime": "9.4.0",
        "proj_db": "/usr/share/proj/proj.db",
        "gdal": "3.4.3",
        "geos": "3.9.2",
        "sqlite": "3.36.0"
    }
    ```
<!-- markdownlint-restore -->

Note: if `revision=exported` is reported instead of the git hash then
the `git` program was not available during compilation of GRASS and
the source code did not contain the `.git/` subdirectory (requires e.g.
to `git clone` the GRASS [software
repository](https://github.com/OSGeo/grass/).)

## Citing GRASS

The GRASS Development Team has invested significant time and effort in
creating GRASS, please cite it when using it for data analysis. The
GRASS [Web site](https://grass.osgeo.org/about/license/) offers
citations in different styles.

## AUTHORS

Michael Shapiro, U.S. Army Construction Engineering Research
Laboratory  
Extended info by Martin Landa, Czech Technical University in Prague,
Czech Republic
