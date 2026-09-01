from pathlib import Path
import matplotlib.pyplot as plt
import re
import unicodedata
from .config import (
    DARK_BACKGROUND,
    LIGHT_TEXT,
    GOLD01,
    RED00,
    GREEN01,
    BLUE01,
    B2HHH_DATA,
    B2HHH_SIM,
    DNVTUPLE,
    DELPHES_01
)

class PlotStyle:
    background = DARK_BACKGROUND
    text = LIGHT_TEXT
    signal = GOLD01
    background_model = BLUE01
    prompt = GREEN01
    displaced = RED00

    @staticmethod
    def apply_global_style() -> None:
        plt.rcParams.update({
            "figure.facecolor": DARK_BACKGROUND,
            "axes.facecolor": DARK_BACKGROUND,
            "axes.edgecolor": LIGHT_TEXT,
            "axes.labelcolor": LIGHT_TEXT,
            "axes.titlecolor": LIGHT_TEXT,
            "xtick.color": LIGHT_TEXT,
            "ytick.color": LIGHT_TEXT,
            "text.color": LIGHT_TEXT,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "savefig.facecolor": DARK_BACKGROUND
        })

    @staticmethod
    def apply_dark_axes_style(fig, ax, title, xlabel, ylabel):
        ax.set_facecolor(DARK_BACKGROUND)
        fig.patch.set_facecolor(DARK_BACKGROUND)
        ax.set_title(title, color=LIGHT_TEXT, pad=12, fontweight='bold')
        ax.set_xlabel(xlabel, color=LIGHT_TEXT)
        ax.set_ylabel(ylabel, color=LIGHT_TEXT)
        ax.tick_params(axis='both', colors=LIGHT_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(LIGHT_TEXT)

    @staticmethod
    def add_dark_colorbar(fig, ax, image, label='Frecuencia'):
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(label, color=LIGHT_TEXT)
        colorbar.ax.yaxis.set_tick_params(color=LIGHT_TEXT)
        plt.setp(colorbar.ax.yaxis.get_ticklabels(), color=LIGHT_TEXT)
        colorbar.outline.set_edgecolor(LIGHT_TEXT)
        return colorbar

    @staticmethod
    def safe_filename(value: str) -> str:
        replacements = {
            "→": "to",
            "⇾": "to",
            "±": "pm",
            "+": "plus",
            "-": "minus"
        }
        normalized = str(value)
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)
        normalized = unicodedata.normalize("NFKD", normalized)
        normalized = normalized.encode("ascii", 'ignore').decode('ascii')
        normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_.")
        return normalized

    @staticmethod
    def _output_directory(
        *,
        data: str, 
        output_dir: str | Path | None
    ):
        if output_dir is not None:
            return Path(output_dir)
        if data in ('data', 'b2hhh_data'):
            return B2HHH_DATA
        elif data in ('sim' or 'b2hhh_sim'):
            return B2HHH_SIM
        elif data == 'dnvtuple':
            return DNVTUPLE
        elif data == 'delphes':
            return DELPHES_01
        raise ValueError(f"data tiene que ser 'data/sim', 'dnvtuple' o 'delphes'.")

    @classmethod
    def save_fig(
        cls,
        fig, 
        filename: str,
        *,
        data: str = "delphes",
        output_dir: str | Path | None = None, 
        dpi=500,
        extension: str = "png"
    ):
        directory = cls._output_directory(data=data, output_dir=output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = extension.removeprefix(".")
        path = directory / f"{cls.safe_filename(filename)}.{suffix}"
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches='tight',
            facecolor=fig.get_facecolor()
        )
        print(f'Guardado en: {path}.')
        return path.resolve()

    @classmethod
    def finalize(
        cls,
        fig,
        *,
        save: bool = False,
        filename: str | None = None,
        data: str = 'delphes',
        output_dir: str | Path | None = None,
        dpi: int = 500,
        show: bool = True
    ) -> Path | None:
        fig.tight_layout()
        saved_path = None
        if save:
            if not filename:
                raise ValueError('filename es obligatorio cuando save=True')
            saved_path = cls.save_fig(
                fig=fig, 
                filename=filename,
                data=data,
                output_dir=output_dir,
                dpi=dpi
            )
        if show:
            plt.show()
        return saved_path

PlotStyle.apply_global_style()