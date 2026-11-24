# Legal

Bathyal is a compatibility layer. It lets game files that a user already has run on a Windows PC, the way
Wine and Proton let Windows programs run on Linux. It is written for interoperability, from public
documentation and from observing how programs behave.

This file says what the repository holds, what it never accepts, and how to report a problem. It is not
legal advice.

## What the repository holds

- Source code under GPL-2.0-only (`LICENSE`), forked from [XCL](https://github.com/ivall/xcl).
  `NOTICE.md` lists the upstream credit, and `THIRD_PARTY.md` lists every outside source with its license.
- Tests that build their inputs by hand (synthetic fixtures), or that read a user's own local files
  through a path the user sets. No test input comes from a game.
- Documentation about file formats and console behaviour, written in our own words.

## What it never holds or accepts

- Game files of any kind: executables, archives, assets, shaders, audio, video, saves, or logs and dumps
  that quote them.
- Keys, decryption code, or code that removes or bypasses copy protection, licence checks or DRM.
  Bathyal reads files that are already unpacked. It has no means to unpack or decrypt a protected package.
- Material from a console SDK or any other source under a non-disclosure agreement, and code or tables
  copied from a project whose license is not compatible with GPL-2.0.
- Links to copies of games, instructions for getting them, or help with running copies the user does
  not own. Issues and pull requests that ask for or offer them are closed and the content is deleted.
- Code that contacts an online service. Bathyal runs offline.

Every pull request goes through the content gate (`scripts/check_content.py`). It fails a change that
adds a game binary, a shader or media container, an archive, a network library, or a commercial game's
name. A maintainer also reviews the change by hand.

## Reporting material that should not be here

If you think this repository holds something that infringes your rights, or anything from the list
above, open an issue titled "Content report" that names the file and the commit. Do not attach the
material. You can also use GitHub's own process (https://docs.github.com/en/site-policy/content-removal-policies).

A maintainer removes the reported material first and reviews it after. Removal from the history follows
when the report is valid.
