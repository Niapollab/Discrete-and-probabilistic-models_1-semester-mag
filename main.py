import logging
from hungarian_method import enumerate_best_assignments_with_prohibitions
from cli import show_menu


def main() -> None:
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    logger = logging.getLogger()

    menu_response = show_menu()
    try:
        for _ in enumerate_best_assignments_with_prohibitions(
            menu_response.appointments, menu_response.prohibitions, logger
        ):
            pass
    except Exception as ex:
        logger.error(ex)


if __name__ == '__main__':
    main()
