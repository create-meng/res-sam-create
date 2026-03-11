import gettext

gettext.bindtextdomain('res_sam', './i18n')
gettext.textdomain('res_sam')
i18n = gettext.gettext