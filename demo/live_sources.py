"""Versioned online source capabilities; no target articles or g scores."""
from urllib.parse import unquote, urlsplit
import re
from copy import deepcopy
from football_research.data_adapters.premier_league_fixture import TEAMS, SCHEDULE_URL, _schedule, _checked_soup
from football_spider.models.contracts import Observation
from football_agent.news_discovery import NewsDiscoveryTools
from football_agent.transport_probe import DiagnosticSession, exception_facts
from football_agent.inquiry_tools import OpenWebTools, ResponseLimit, public_url
from football_agent.inquiry_transport import request_header_profile
import requests
import time


class LivePageSession(DiagnosticSession):
    """8 MB bounded CMS pages, retaining verified TLS and no-netrc routing."""
    def get(self, url, **kwargs):
        kwargs["headers"] = request_header_profile(url)["headers"]
        response = requests.Session.get(self, url, stream=True, **kwargs)
        started, chunks, size = time.monotonic(), [], 0
        try:
            for chunk in response.iter_content(16384):
                size += len(chunk)
                if size > 8_000_000 or time.monotonic()-started > 25:
                    raise ResponseLimit("Online page response limit")
                chunks.append(chunk)
            response._content = b"".join(chunks)
            response._content_consumed = True
            return response
        except requests.RequestException as exc:
            self.failure = exception_facts(exc)
            response.close()
            raise


class LiveNewsTools(NewsDiscoveryTools):
    """Regional RSS; every transport attempt is separately archived by caller."""
    SEARCH_ENDPOINT = "https://cn.bing.com/search"

    def _delegate(self, route):
        return OpenWebTools(self.archive, self.allowed_hosts,
            session_factory=lambda: LivePageSession(route), clock=self.clock, extraction_version=self.extraction_version)

    def configuration(self):
        return {"schema": "footballmind-live-source-transport-v1", "allowed_hosts": self.allowed_hosts,
            "index_urls": self.index_urls, "projection_version": self.projection_version,
            "search_endpoint": self.SEARCH_ENDPOINT, "max_response_bytes": 8_000_000,
            "response_read_elapsed_limit_seconds": 25, "route": self.default_route,
            "route_overrides": dict(self.route_overrides),
            "request_attempts_per_action": 1, "automatic_redirects": False,
            "search_route": getattr(self, "search_route", self.default_route),
            "live_body_adapter": "footballmind-live-webflow-news-body-v1",
            "additional_live_body_adapters": ["fortuna-visible-news-body-v1", "same-page-og-semantic-article-body-v1", "footballmind-live-venezia-same-page-body-v1", "footballmind-live-dutch-registered-body-v1"],
            "live_directory_adapter": "footballmind-live-literal-directory-cards-v1",
            "additional_live_directory_adapters": ["footballmind-live-italian-observed-directory-cards-v1", "footballmind-live-ceuta-observed-news-cards-v1"],
            "tls_verification": True, "authentication": "none_no_netrc"}

    def prepare(self, candidate, public):
        # The generic tool normalizes https://host to https://host/ before its
        # known-link check. Normalize the already authorized set the same way;
        # otherwise a registered club homepage fails before its first request.
        canonical = deepcopy(public)
        canonical["task"]["seed_urls"] = [public_url(u, self.allowed_hosts) for u in public["task"]["seed_urls"]]
        for doc in canonical["documents"]:
            for link in doc.get("links", []):
                try:
                    link["url"] = public_url(link["url"], self.allowed_hosts)
                except ValueError:
                    pass  # An unapproved URL still fails candidate public_url validation.
        prepared = super().prepare(candidate, canonical)
        if prepared["tool"] == "search_web":
            prepared["endpoint"] = self.SEARCH_ENDPOINT
        return prepared

    def project(self, observation, prepared):
        document = super().project(observation, prepared)
        if prepared.get("tool") == "read_url":
            from .live_article_projection import project_live_article, project_live_directory
            raw = self.archive.read_body(observation)
            if document.get("url") in self.index_urls:
                document = project_live_directory(document, raw, observation, self.allowed_hosts)
                from .live_official_directory_cards import project_official_directory_cards
                document = project_official_directory_cards(document, raw, observation, self.allowed_hosts)
            else:
                document = project_live_article(document, raw, observation)
        return document

    def execute(self, prepared, *, run_id, request_id):
        if prepared.get("tool") != "search_web":
            return super().execute(prepared, run_id=run_id, request_id=request_id)
        if (set(prepared) != {"tool", "endpoint", "params", "source"}
                or prepared["endpoint"] != self.SEARCH_ENDPOINT
                or set(prepared["params"]) != {"q", "format"}
                or prepared["params"]["format"] != "rss" or prepared["source"] != "open_search"):
            raise ValueError("Unexpected online RSS request")
        returned = self._delegate(getattr(self, "search_route", self.default_route)).execute(prepared, run_id=run_id, request_id=request_id)
        returned["document"] = self.project(Observation.model_validate(returned["observation"]), prepared)
        return returned

TEAMS = {**TEAMS, "57": ("考文垂", "Coventry City")}
DIRECTORIES = {
    "Arsenal": "https://www.arsenal.com/news",
    "Leeds United": "https://www.leedsunited.com/en/news",
    "Liverpool": "https://www.liverpoolfc.com/news/category/men",
    "Aston Villa": "https://www.avfc.co.uk/",
    "Brentford": "https://www.brentfordfc.com/en/news/listing/all-news",
    "Ipswich Town": "https://www.itfc.co.uk/",
    "Fulham": "https://www.fulhamfc.com/news/",
    "Chelsea": "https://www.chelseafc.com/en",
    "Crystal Palace": "https://www.cpfc.co.uk/news/",
    "Nottingham Forest": "https://www.nottinghamforest.co.uk/news/",
    # Public generic directories independently discovered 2026-10-19. No
    # search-discovered article is seeded into an Agent menu.
    "Tottenham Hotspur": "https://www.tottenhamhotspur.com/",
    "Coventry City": "https://www.ccfc.co.uk/news/",
    # Generic club directories observed from official sites, 2026-10-19.
    "Parma": "https://www.parmacalcio1913.com/en/main-news/",
    "Torino": "https://www.torinofc.it/en/news/all",
    # Official club directories verified with direct 200 responses on
    # 2026-10-24; exact paths avoid automatic redirect following.
    "Augsburg": "https://www.fcaugsburg.de/",
    "Union Berlin": "https://www.fc-union-berlin.de/de",
    # Generic official directories verified with archived HTTP 200 on 2026-10-25.
    "GAIS": "https://www.gais.se/",
    "IFK Göteborg": "https://ifkgoteborg.se/nyheter/nyheter-notiser/",
    "AV Alta FC": "https://www.avaltafc.com/news/",
    "Corpus Christi FC": "https://www.corpuscfc.com/",
    # J 联赛官方目录，2026-10-25 直接 HTTP 200 核验（标题为俱乐部官网自身）：
    #   https://www.cerezo.jp/news/ 服务端渲染，含 /news/<date> 文章永久链接；
    #   https://jefunited.co.jp/news/list 可达但列表由 JS 渲染，服务端 HTML 无文章链接
    #   （保留域名用于后续 CMS/JS 适配与搜索结果域白名单，不影响目录读取）。
    "Cerezo Osaka": "https://www.cerezo.jp/news/",
    "JEF United Ichihara-Chiba": "https://jefunited.co.jp/news/list",
    # 欧洲联赛官方新闻目录：2026-10-25 逐个只读核验（HTTP 200、官网标题、服务端渲染且
    # 含 ≥5 条文章形状链接）。键为运行时 canonical 英文名（ESPN displayName 或 TEAMS 映射）。
    # 未采用的：Crystal Palace/Sunderland（服务端 HTML 无文章链接）、Hull City、Bayern Munich
    #   （传输错误）、Athletic Club（403 挑战页）、Celta（404）、Hoffenheim（1 条）、
    #   AFC Bournemouth（2 条）、Bayer Leverkusen 以外的德甲站点。
    "Newcastle United": "https://www.newcastleunited.com/en/news",
    "Manchester United": "https://www.manutd.com/en/news",
    "Internazionale": "https://www.inter.it/en/news",
    "Inter Milan": "https://www.inter.it/en/news",
    "Fiorentina": "https://www.acffiorentina.com/en/news",
    "Atalanta": "https://en.atalanta.it/news",
    "Frosinone": "https://www.frosinonecalcio.com/news/",
    "Genoa": "https://genoacfc.it/news/",
    "Venezia": "https://www.veneziafc.it/news",
    "热那亚": "https://genoacfc.it/news/",
    "威尼斯": "https://www.veneziafc.it/news",
    "Lecce": "https://uslecce.it/news/",
    "Juventus": "https://www.juventus.com/en/news/",
    "Udinese": "https://www.udinese.it/news",
    "AC Milan": "https://www.acmilan.com/en/news",
    "Freiburg": "https://www.scfreiburg.com/",
    "SC Freiburg": "https://www.scfreiburg.com/",
    "Bayer Leverkusen": "https://www.bayer04.de/de-de",
    "Getafe": "https://www.getafecf.com/Noticias",
    "Real Betis": "https://www.realbetisbalompie.es/noticias/",
    "Real Sociedad": "https://www.realsociedad.eus/es/noticias",
    "Levante": "https://www.levanteud.com/es/noticias",
    "Barcelona": "https://www.fcbarcelona.com/en/news",
    "Real Madrid": "https://www.realmadrid.com/en-US",
    "Monaco": "https://www.asmonaco.com/en/news",
    "Lille": "https://www.losc.fr/actualites",
    # 荷甲/比甲：2026-10-25 逐个核验（HTTP 200、服务端渲染、≥5 条文章形状链接）。
    # 未采用：PSV Eindhoven（/nieuws 404、首页 0 条）、Sparta Rotterdam（仅 eSports 子栏目）、
    #   Club Brugge（429 Vercel 挑战页）、Anderlecht（202 挑战页）。
    "Fortuna Sittard": "https://fortunasittard.nl/nieuws",
    "Excelsior": "https://excelsiorrotterdam.nl",
    "Ajax Amsterdam": "https://www.ajax.nl/",
    "Feyenoord": "https://www.feyenoord.com/nl/news",
    "AZ Alkmaar": "https://www.az.nl/inside-az",
    "FC Twente": "https://fctwente.nl/nieuws",
    "NEC Nijmegen": "https://www.nec-nijmegen.nl/nieuws.htm",
    "FC Groningen": "https://www.fcgroningen.nl/nieuws/",
    "Racing Genk": "https://www.krcgenk.be/nl/news",
    "Standard Liege": "https://standard.be/fr/news",
    "Cercle Brugge KSV": "https://cerclebrugge.be/home",
    "Union St.-Gilloise": "https://rusg.brussels/en/actualites",
    "Cádiz": "https://www.cadizcf.com/noticias",
    "Ceuta": "https://adceutafc.com/noticias/",
}

# 日文队名：英文队名在日文来源上的召回极差，且 J 联赛没有英文官方目录。
# 仅登记已核验的官方写法；未登记的球队不会继承其他球队的名称。
JP_NAMES = {
    "Cerezo Osaka": "セレッソ大阪",
    "JEF United Ichihara-Chiba": "ジェフユナイテッド千葉",
}

# Source-navigation aliases only; these never merge fixture IDs or assert a
# player's status. The competition/squad admission below still applies.
SOURCE_NAME_ALIASES = {"热那亚": "Genoa", "威尼斯": "Venezia"}

# Explicit association/squad scope for every registered club. A display name
# alone cannot distinguish Liverpool (England) from Liverpool (Uruguay), or a
# club's men's side from its women's side. Cup/UEFA eligibility is evaluated
# against these declarations, never inferred from an article title or URL.
CLUB_ASSOCIATION_GROUPS = {
    "ENG": ("Arsenal", "Leeds United", "Liverpool", "Aston Villa", "Brentford", "Ipswich Town",
        "Fulham", "Chelsea", "Crystal Palace", "Nottingham Forest", "Tottenham Hotspur",
        "Coventry City", "Newcastle United", "Manchester United"),
    "ITA": ("Parma", "Torino", "Internazionale", "Inter Milan", "Fiorentina", "Atalanta", "Frosinone",
        "Genoa", "Venezia", "热那亚", "威尼斯", "Lecce", "Juventus", "Udinese", "AC Milan"),
    "GER": ("Augsburg", "Union Berlin", "Freiburg", "SC Freiburg", "Bayer Leverkusen"),
    "ESP": ("Getafe", "Real Betis", "Real Sociedad", "Levante", "Barcelona", "Real Madrid", "Cádiz", "Ceuta"),
    "FRA": ("Monaco", "Lille"),
    "NED": ("Fortuna Sittard", "Excelsior", "Ajax Amsterdam", "Feyenoord", "AZ Alkmaar", "FC Twente", "NEC Nijmegen", "FC Groningen"),
    "BEL": ("Racing Genk", "Standard Liege", "Cercle Brugge KSV", "Union St.-Gilloise"),
    "SWE": ("GAIS", "IFK Göteborg"),
    "USA": ("AV Alta FC", "Corpus Christi FC"),
    "JPN": ("Cerezo Osaka", "JEF United Ichihara-Chiba"),
}
CLUB_SOURCE_SCOPE = {name: {"association": association, "squad": "men_senior"}
    for association, names in CLUB_ASSOCIATION_GROUPS.items() for name in names}
EUROPEAN_ASSOCIATIONS = frozenset({"ENG", "ITA", "GER", "ESP", "FRA", "NED", "BEL", "SWE"})


def competition_source_scope(competition):
    """Resolve registered competition families, not arbitrary identity aliases."""
    name = str(competition or "").strip().casefold()
    if not name:
        return {"association": None, "squad": None, "basis": "legacy_missing_competition"}
    if re.search(r"women|female|fémin|feminin|femen|vrouwen|frauen|ladies|nwsl|wsl|女", name) or name in {"liga f", "première ligue", "northern super league"}:
        squad = "women"
    elif re.search(r"youth|under[- ]?\d|\bu[- ]?(?:17|19|20|21|23)\b|青年|少年", name):
        squad = "youth"
    else:
        squad = "men_senior"
    if "uefa" in name or name in {"欧冠", "欧罗巴", "欧联杯", "欧协联"}:
        return {"association": "UEFA", "squad": squad, "basis": "explicit_european_club_competition"}
    groups = [
        ("ENG", ("premier league", "english premier league", "英超", "英冠", "英甲", "英乙", "english championship", "efl championship", "carabao cup", "efl trophy", "fa cup", "fa community shield", "national league")),
        ("ITA", ("意甲", "意乙", "serie a", "serie b", "coppa italia", "italian cup")),
        ("GER", ("德甲", "德乙", "bundesliga", "german cup", "dfb-pokal")),
        ("ESP", ("西甲", "西乙", "laliga", "la liga", "copa del rey", "spanish cup")),
        ("FRA", ("法甲", "法乙", "ligue 1", "ligue 2", "coupe de france", "french cup")),
        ("NED", ("荷甲", "eredivisie", "knvb", "keuken kampioen")),
        ("BEL", ("比甲", "belgian", "jupiler", "croky cup")),
        ("SWE", ("瑞超", "allsvenskan", "svenska cupen")),
        ("USA", ("美职联", "美职", "mls", "usl", "us open cup", "u.s. open cup")),
        ("JPN", ("日职", "日乙", "j1 league", "j2 league", "j.league", "japan", "emperor's cup", "ルヴァン")),
        ("URU", ("liga auf uruguaya", "uruguayan", "乌拉圭")),
        ("SCO", ("苏超", "spfl", "scottish", "scotland")),
        ("ARG", ("阿甲", "argentine", "argentina")),
        ("BRA", ("巴甲", "巴乙", "brazil", "brasileiro")),
        ("CHN", ("中超", "中甲", "chinese", "china league")),
    ]
    # More specific foreign families must win over generic words such as Serie A.
    if "austrian" in name:
        return {"association": "AUT", "squad": squad, "basis": "explicit_competition_family"}
    if "brazil" in name or "brasileiro" in name or name in {"巴甲", "巴乙"}:
        return {"association": "BRA", "squad": squad, "basis": "explicit_competition_family"}
    for association, aliases in groups:
        # "Russian Premier League" must not match England solely because it
        # contains "Premier League". ESPN round labels follow a comma.
        matched = (any(name == alias or name.startswith(alias + ",") for alias in aliases)
                   if association == "ENG" else any(alias in name for alias in aliases))
        if matched:
            return {"association": association, "squad": squad, "basis": "explicit_competition_family"}
    return {"association": None, "squad": squad, "basis": "competition_scope_unregistered"}


def club_source_admission(name, competition):
    declared = CLUB_SOURCE_SCOPE.get(name)
    scope = competition_source_scope(competition)
    if name not in DIRECTORIES or declared is None:
        return {"allowed": False, "reason": "club_directory_or_scope_unregistered", "club_scope": declared, "competition_scope": scope}
    if scope["basis"] == "legacy_missing_competition":
        return {"allowed": True, "reason": "legacy_name_only_compatibility_not_runtime_identity_proof", "club_scope": declared, "competition_scope": scope}
    if scope["squad"] != declared["squad"]:
        return {"allowed": False, "reason": "club_squad_scope_mismatch", "club_scope": declared, "competition_scope": scope}
    association = scope["association"]
    allowed = association == declared["association"] or association == "UEFA" and declared["association"] in EUROPEAN_ASSOCIATIONS
    return {"allowed": allowed, "reason": "association_and_squad_match" if allowed else "club_competition_association_mismatch_or_unknown",
            "club_scope": declared, "competition_scope": scope}

# Exact public news hosts. Search results must still pass URL/host validation;
# this registry does not assert reliability, relevance, or information value.
SEARCH_HOSTS = ["www.bbc.com", "www.bbc.co.uk", "www.skysports.com", "www.theguardian.com",
                "www.uefa.com", "www.fifa.com", "www.espn.com", "football-italia.net",
                "ge.globo.com", "www.espn.com.br", "www.kicker.de", "www.marca.com",
                "www.bundesliga.com", "www.laliga.com", "www.legaseriea.it", "ligue1.com",
                "www.jleague.jp", "eredivisie.nl", "nos.nl", "www.vi.nl", "spfl.co.uk"]


def competition_navigation(competition):
    """Navigation language/site hints, never evidence-topic or value filters."""
    name = str(competition or "").casefold()
    groups = [
        (("德甲", "德乙", "bundesliga", "german cup"), ("bundesliga.com", "kicker.de"), "Fußball Vorschau Nachrichten"),
        (("西甲", "西乙", "laliga", "la liga", "copa del rey"), ("laliga.com", "marca.com"), "fútbol previa noticias"),
        (("liga mx", "墨超"), ("ligamx.net", "espn.com.mx"), "fútbol previa noticias"),
        (("意甲", "意乙", "serie a", "serie b"), ("legaseriea.it", "football-italia.net"), "calcio anteprima notizie"),
        (("法甲", "法乙", "ligue 1", "ligue 2"), ("ligue1.com", "lequipe.fr"), "football avant-match actualités"),
        (("荷甲", "eredivisie", "knvb", "keuken"), ("eredivisie.nl", "vi.nl"), "Eredivisie voetbal voorbeschouwing nieuws"),
        (("比甲", "belgian", "jupiler"), ("sporza.be", "proleague.be"), "voetbal wedstrijd nieuws"),
        (("中超", "中甲", "chinese super", "china league"), ("thecfa.cn", "sports.sina.com.cn"), "足球 赛前 新闻"),
        (("日职", "日乙", "j1 league", "j2 league", "j.league"), ("jleague.jp",), "サッカー 試合前 ニュース"),
        (("葡超", "portuguese", "primeira liga"), ("ligaportugal.pt", "abola.pt"), "futebol prévia notícias"),
        (("巴甲", "巴乙", "brasileiro"), ("ge.globo.com", "espn.com.br"), "futebol prévia notícias"),
        (("瑞超", "allsvenskan"), ("svenskfotboll.se", "svt.se/sport"), "fotboll inför match nyheter"),
        (("土超", "turkish", "super lig"), ("tff.org", "trtspor.com.tr"), "futbol maç öncesi haber"),
        (("k league", "韩职"), ("kleague.com",), "축구 경기 프리뷰 뉴스"),
        (("usl league one",), ("uslleagueone.com", "espn.com"), "soccer match preview news"),
    ]
    for aliases, sites, language in groups:
        if any(alias in name for alias in aliases):
            return list(sites), language
    return ["espn.com", "bbc.com/sport"], "football match preview news"


def bind_identity(manifest, returned, archive, at):
    from .live_g import utc
    fixture = manifest["market_initialization"]["fixture"]
    names, aliases = [], {}
    if fixture["competition_name"] not in {"英超", "英格兰超级联赛", "Premier League", "PL"}:
        raise ValueError("在线实验目前只支持英超")
    for side in ("home", "away"):
        ident = fixture[side+"_team_id"]
        mapped = TEAMS.get(ident.removeprefix("spftll:team:")) if ident.startswith("spftll:team:") else None
        if not mapped or mapped[0] != fixture[side+"_team_name"]:
            raise ValueError("未知或冲突的球队身份映射")
        names.append(mapped[1])
        aliases[side] = list(mapped)
    observation = Observation.model_validate(returned["observation"])
    soup = _checked_soup(observation, archive.read_body(observation), at)
    extracted = _schedule(soup, tuple(names)) if soup else {"status": "UNCONFIRMED"}
    if extracted["status"] != "EXTRACTED" or utc(extracted["canonical"]["kickoff"]) != utc(fixture["kickoff"]):
        raise ValueError("来源与官方赛程的球队顺序或开球时间不一致")
    return {"schema": "footballmind-live-october-identity-v1", "status": "CONFIRMED",
            "canonical": extracted["canonical"], "aliases": aliases, "verified_at": at,
            "observation": returned["observation"], "parser_scope": "2026-27 October official PL article",
            "source_event_id_is_local_namespace": True}


def profile(binding):
    names = [binding["canonical"][s+"_name"] for s in ("home", "away")]
    competition_name = binding["canonical"].get("competition_name")
    admissions = [{"team_name":n, **club_source_admission(n, competition_name)} for n in names]
    seeds = list(dict.fromkeys(DIRECTORIES[r["team_name"]] for r in admissions if r["allowed"]))
    unknown = [r["team_name"] for r in admissions if not r["allowed"]]
    competition = str(competition_name or "").casefold()
    # Exact official league indexes independently fetched/archived HTTP 200 on
    # 2026-10-25. They supply navigation when a club directory is not registered;
    # unlike a club seed, their article metadata still needs team binding.
    league_seed = None
    if unknown:
        if competition in {"premier league", "english premier league", "英超"}:
            league_seed = "https://www.premierleague.com/en/news"
        elif competition in {"german bundesliga", "bundesliga", "德甲"}:
            league_seed = "https://www.bundesliga.com/en/bundesliga/news"
        elif competition in {"spanish laliga", "la liga", "laliga", "西甲"}:
            league_seed = "https://www.laliga.com/en-GB/news"
        elif competition in {"苏超", "spfl premiership", "scottish premiership"}:
            league_seed = "https://spfl.co.uk/news"
        elif competition in {"日职", "j1 league", "japanese j.league", "japanese j1 league"}:
            league_seed = "https://www.jleague.jp/j1/news/"
    if league_seed:
        seeds.append(league_seed)
    return {"schema": "footballmind-live-source-profile-v3", "seed_urls": seeds,
            "allowed_hosts": sorted({urlsplit(u).hostname for u in seeds} | set(SEARCH_HOSTS)),
            "unknown_teams": unknown,
            "club_scope_audit": admissions,
            "source_scope_version": "explicit-club-association-squad-v1",
            "league_wide_seed_urls": [league_seed] if league_seed else [],
            "discovery_mode": "club_directories_with_search_fallback" if seeds else "public_match_search",
            "basis": "explicit_club_directories_and_bounded_public_news_search_not_article_value"}


def discovery_queries(canonical):
    """Fixture and competition specific navigation; no fixed German fallback."""
    canonical = dict(canonical)
    for side in ("home", "away"):
        name = canonical[side+"_name"]
        if name in SOURCE_NAME_ALIASES and club_source_admission(name, canonical.get("competition_name"))["allowed"]:
            canonical[side+"_name"] = SOURCE_NAME_ALIASES[name]
    home, away = canonical["home_name"], canonical["away_name"]
    home = "FC Augsburg" if home == "Augsburg" else home
    pair = '"'+home+'" "'+away+'"'
    competition = canonical.get("competition_name", "football")
    date = canonical["kickoff"][:10]
    sites = {
        "德甲": ["kicker.de", "bundesliga.com"], "German Bundesliga": ["kicker.de", "bundesliga.com"],
        "英超": ["bbc.com/sport", "skysports.com"], "Premier League": ["bbc.com/sport", "skysports.com"],
        "英冠": ["bbc.com/sport", "skysports.com"], "瑞超": ["svenskfotboll.se", "svt.se/sport"],
        "Swedish Allsvenskan": ["svenskfotboll.se", "svt.se/sport"],
        "USL League One": ["uslleagueone.com", "espn.com"],
        "西甲": ["laliga.com", "marca.com"], "意甲": ["legaseriea.it", "football-italia.net"],
        "法甲": ["ligue1.com", "lequipe.fr"], "中超": ["thecfa.cn", "sports.sina.com.cn"],
        # J 联赛：2026-10-25 核验 www.jleague.jp 为官方站点；英语新闻源对该赛事召回极差，
        # 因此日文查询与官方站点优先，而不是回退到 espn/bbc（见下）。
        "日职": ["jleague.jp"], "日乙": ["jleague.jp"],
        "J1 League": ["jleague.jp"], "J2 League": ["jleague.jp"],
        "Japanese J.League": ["jleague.jp"], "Japanese J1 League": ["jleague.jp"],
    }.get(competition)
    navigation_sites, local_context = competition_navigation(competition)
    sites = sites or navigation_sites
    japanese = home in JP_NAMES and away in JP_NAMES
    queries = [pair+" soccer "+date+" preview team news"]
    if japanese:
        # 日文表述放在第 2 位：运行时只取前 5 条，召回最差的英文查询不能挤掉它。
        queries.append(JP_NAMES[home]+" "+JP_NAMES[away]+" 試合前 ニュース スタメン")
    queries.append(pair+" "+competition+" "+local_context)
    # Club-specific queries can find context not naming both sides in a title.
    # 日文赛事已有官方俱乐部目录 seed，标题里的英文队名对日文来源无效，故不生成。
    if not japanese:
        for name in (canonical["home_name"], away):
            if club_source_admission(name, competition)["allowed"]:
                queries.append("site:"+urlsplit(DIRECTORIES[name]).hostname+' "'+name+'" '+local_context)
    queries.extend("site:"+site+" "+pair+" "+local_context for site in sites)
    queries.append(pair+" soccer injury availability")
    return list(dict.fromkeys(queries))[:5]


def search_team_binding(title, url, fixture, entries):
    """Visible team/context admission only; does not assert body usefulness."""
    from .live_g import _team_tokens
    if urlsplit(url).hostname in {urlsplit(entry).hostname for entry in entries}:
        return True
    visible = str(title)+" "+unquote(urlsplit(url).path).replace("-", " ")
    observed = _team_tokens(visible)
    teams = set()
    for side in ("home", "away"):
        for name in [fixture.get(side+"_name"), *(fixture.get(side+"_aliases") or [])]:
            teams.update(_team_tokens(name))
            if name in SOURCE_NAME_ALIASES and club_source_admission(name, fixture.get("competition_name"))["allowed"]:
                teams.update(_team_tokens(SOURCE_NAME_ALIASES[name]))
            # Whitespace word tokenization cannot find a Chinese/Japanese team
            # name embedded in an otherwise continuous headline. Match only an
            # explicit fixture alias, never derive an alias from the article.
            if isinstance(name, str) and len(name) >= 2 and re.search(r"[\u3040-\u30ff\u3400-\u9fff]", name):
                if name.casefold() in visible.casefold():
                    return True
    return bool(observed & teams)


def article_link_reason(url, title, entries, seen, *, search=False):
    """Admit observed article-shaped URLs, including paths outside /news/."""
    parsed = urlsplit(url)
    path = unquote(parsed.path).rstrip("/")
    # Short CJK news headlines contain more information per character than a
    # Latin navigation label. Article URL structure is still required below.
    cjk_count = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", str(title)))
    if url in seen or url in entries or (len(str(title).strip()) < 12 and cjk_count < 3):
        return "duplicate_index_or_short_navigation_label"
    # These two registered club directories serve the men's source profile.
    # Restored card names explicitly identify separate Italian women's/youth
    # teams; preserve their observed menu rows but do not mix squad identities.
    if parsed.hostname in {"genoacfc.it", "www.veneziafc.it"} and re.search(
            r"\b(?:fem\.?|femminile|femminili|primavera)\b|\bsettore\s+giovanile\b", str(title), re.I):
        return "directory_other_squad_label"
    # Observed Torino menu categories are listings, not individual articles.
    # Keeping their labels in the audit must not suppress the search fallback.
    if urlsplit(url).hostname == "www.torinofc.it" and path in {
            "/en/news/all", "/en/news/prima-squadra", "/en/news/prima-squadra-femminile",
            "/en/news/biglietteria", "/en/news/settore-giovanile", "/en/news/club"}:
        return "observed_news_category_not_article"
    if path.startswith("/nyheter/") and path.rsplit("/", 1)[-1] in {"kalender", "nyheter-notiser"}:
        return "observed_news_category_not_article"
    # News taxonomies differ across leagues. Accept an explicit article under
    # any observed news section; section/category/listing roots stay navigation.
    sections = {"news", "article", "articles", "main-news", "meldungen", "meldung", "nyheter",
                "nieuws", "noticias", "notizie", "actualites", "actualités", "actu",
                "nachrichten", "haber", "haberler", "noticia", "notizia", "inside-az"}
    parts = [p.casefold() for p in path.split("/") if p]
    listing = {"all", "list", "listing", "archive", "archives", "category", "categories",
               "kalender", "nyheter-notiser", "nieuwsoverzicht", "news", "noticias"}
    if parts and parts[-1] in listing:
        return "observed_news_category_not_article"
    if any(p in sections for p in parts[:-1]) and parts[-1] not in sections:
        return None
    official = parsed.hostname in {urlsplit(entry).hostname for entry in entries}
    if official and re.search(r"/20\d{2}/\d{1,2}/(?:\d{1,2}/)?[^/]+$", path):
        return None
    # Broad sports publishers use /football/DATE/slug or root article slugs.
    # Only public search results can enter by this rule; directory navigation
    # does not become a body candidate merely because it has a long label.
    slug = path.rsplit("/", 1)[-1]
    if search and (slug.count("-") >= 2 or any(token in path for token in
            ("/story/", "/stories/", "/preview", "/team-news", "/injur", "/lineup", "/squad"))):
        return None
    return "outside_article_link_shape"
