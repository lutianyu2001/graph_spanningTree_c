/*****************************************************************
Programmer:  Lu Tianyu (Sky)
Student ID: 1930026092
Date: 2021/12/03 01:35
Language: C flavoured C++ (C Based, only use some C++ classes)
Task Name: Design and Analysis of Algorithms - Individual Assignment
Purpose: Write a program to return those artifacts from a given weighted, undirected graph:
    1. A Depth-First Search (DFS) Tree
    2. All its Articulation Points (AP) and Biconnected Components (BC)
    3. A Minimum Spanning Tree (MST), using Kruskal's Algorithm
    4. The Shortest Path Tree (SPT), using Dijkstra's Algorithm
 *****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>  // boolean is included in c++, actually no need to include this library
#include <list>
#include <stack>
using namespace std;

typedef struct{
    int start;
    int end;
    int weight;
}edge;

typedef struct graphVertex{
    struct graphVertex *leftFirstChild;
    int name; // vertex name, in our case is just the vertex number
    struct graphVertex *nextSibling;
}DFSvertex;

typedef struct{
    int info[2]; // info[0] - vexnum (number of vertexes), info[1] - edgnum (number of edges)
    edge *edges; // malloc needed, 1-d array for storing edges
    int **adjM; // malloc needed, 2-d array for storing adjacency Matrix

    DFSvertex *DFSforest; // Depth-First Search Forest using the "Binary Tree Representation" of forest
    // Reference for the binary tree representation of forest: http://data.biancheng.net/view/198.html

    edge *MST; // malloc needed, array for storing edges in the Minimum Spanning Tree
    int MSTinfo; // showing how many edges are there in the MST

    edge *SPT; // malloc needed, array for storing edges in the Smallest Path tree
    int SPTinfo; // showing how many edges are there in the SPT
    int *SPTdist; // malloc needed, 1-d array for storing distance of each vertex in SPT
}graph;

#define INF 2147483647 // use maximum value of signed integer as infinity
#define swap(a, b) { typeof(a) tmp = (a); (a) = (b); (b) = tmp; }
#define min(a, b) { ((a) < (b)) ? (a) : (b) }

int singleProcedure();

char *menu_inputFileName();
FILE* openFile(const char *mode);
void importGraph(FILE *pfile, graph *G);
void initMSTandSPT(graph *G);
void debug_showAdjM(graph *G);
void destroyGraph(graph *G);
void destroyDFSforest(DFSvertex **currentVertex);

edge* createEdgeArray(int length);
bool* createBoolArray(int length, int initValue);
int* createIntArray(int length, int initValue, bool initWithI);
void debug_printBoolArray(int length, bool *A);
void debug_printIntArray(int length, int *A);

void generateDFSforest(graph *G);
int adjM_nextVertex(graph *G, int vertex, int lastVertex);
void generateDFStree(graph *G, int v, bool *visit, DFSvertex *currentVertex);
void showDFSforest(DFSvertex *currentVertex, int father);

void artPtBiComp(graph *G);
void reArtPt(graph *G, int v, bool *visit, int *d, int *pred, int *low,
             int *time, stack<edge> *s, bool *artPtMark, edge *biComp, int *biCompInfo);
edge* createEdge(int start, int end, int weight);
//bool compareEdge(edge *E, edge *edge2);  // FOR DEBUG
bool isEdge(edge E, int start, int end, int weight);
void setBiCompDelimiter(edge *biComp, int *biCompInfo);
void sortEdgeArrayByVertex(edge *A, int left, int right);

void kruskalMST(graph *G);
void addToMST(graph *G, int edgeIdx);
//void sortMST(graph *G);  // ABANDONED
void showMST(graph *G);

int findSet(int *father, int x);
void unionFind(int *father, int *rank, int a, int b);
int partitionEdge(edge *e, int left, int right, int judge);
void quickSortEdge(edge *e, int left, int right, int judge);
void quickSortEdgeVertex(graph *G);

void generateSPT(graph *G);
void dijkstra(graph *G, int start, int *pred, int *dist);
void addToSPT(graph *G, int start, int end);
//void sortSPT(graph *G);  // ABANDONED
void showSPT(graph *G);


int main() {
    while(singleProcedure() != -1);
    return 0;
}

int singleProcedure() {
    const char *fileMode = "r";
    FILE *pfile = openFile(fileMode);
    if(pfile == NULL) return -1;
    graph G;
    importGraph(pfile, &G);

    generateDFSforest(&G);
    showDFSforest(G.DFSforest, -1);
    printf("\n");

    artPtBiComp(&G);

    kruskalMST(&G);
    showMST(&G);

    generateSPT(&G);
    showSPT(&G);

    destroyGraph(&G);
    return 0;
}


// ---------------------------------------------------------------------------------------------
//                        --- I. Initialisation: Load Graph from file ---
//                               Entrance: openFile() -> importGraph()
// ---------------------------------------------------------------------------------------------

FILE* openFile(const char *mode) {
    FILE *pfile = NULL;
    char* fname = menu_inputFileName();
    if(fname == NULL) return NULL; // pass the NULL to upper-level function
    pfile = fopen(fname, mode);
    while(!pfile){
        printf("ERROR: Cannot open/create \"%s\", please check again.\n",fname);
        fname = menu_inputFileName();
        if(fname == NULL) return NULL; // pass the NULL to upper-level function
        pfile = fopen(fname, mode);
    }
    return pfile;
}

char* menu_inputFileName() {
    printf("***************\nInput the file name:\n");
    printf("(including suffix names like \".txt\", the whole name will <= 50 characters)\n");
    static char fname[51]; // use static to get a persistent variable
    fflush(stdin); // use fflush() to flush the standard IO stream
    char *strPtr = gets(fname); // gets returns NULL when received EOF
    fflush(stdin);
    if(strPtr == NULL) return NULL; // pass the NULL to upper-level function
    return fname;
}

void importGraph(FILE *pfile, graph *G) {
    // >>> Import Basic Info: graph size >>>
    int i = -1;
    while(++i < 2) fscanf(pfile, "%d%*c", &((*G).info[i])); // use %*c to absorb \n

    // >>> Import Graph Edge List >>>
    (*G).edges = (edge*) malloc((*G).info[1] * sizeof(edge));
    i = -1;
    while(++i < (*G).info[1]){
        fscanf(pfile, "%d,%d,%d%*c",
               &(*G).edges[i].start,&(*G).edges[i].end,&(*G).edges[i].weight);
        if((*G).edges[i].start > (*G).edges[i].end) swap((*G).edges[i].start, (*G).edges[i].end)
        // ensure the start vertex must be smaller than the end vertex
    }

    // >>> Import Graph by Adjacency Matrix >>>
    // dynamic allocating an 2-d array for storing the matrix
    (*G).adjM = (int**) malloc((*G).info[0] * sizeof(int*));
    for(i=0; i<(*G).info[0]; i++) (*G).adjM[i] = (int*) malloc((*G).info[0] * sizeof(int));
    // fill the matrix with 0
    for(int j=0; j<(*G).info[0]; j++) for(int k=0; k<(*G).info[0]; k++) (*G).adjM[j][k] = 0;
    // fill in the matrix with edges
    for(i=0; i<(*G).info[1]; i++){
        (*G).adjM[(*G).edges[i].start][(*G).edges[i].end] = (*G).edges[i].weight;
        (*G).adjM[(*G).edges[i].end][(*G).edges[i].start] = (*G).edges[i].weight;
    }
    //debug_showAdjM(G);  // FOR DEBUG
    initMSTandSPT(G);
    fclose(pfile);
    pfile = NULL;
}

void initMSTandSPT(graph *G){
    // the number of edges in MST and SPT must not exceed vexnum - 1 ((*G).info[0] - 1)
    (*G).MST = createEdgeArray((*G).info[0] - 1);
    (*G).MSTinfo = 0;

    (*G).SPT = createEdgeArray((*G).info[0] - 1);
    (*G).SPTinfo = 0;
    (*G).SPTdist = createIntArray((*G).info[0], INF, 0);
}

void debug_showAdjM(graph *G){
    printf("-------------------\n");
    for(int j=0; j<(*G).info[0]; j++){
        for(int k=0; k<(*G).info[0]; k++){
            printf("%d ", (*G).adjM[j][k]);
        }
        printf("\n");
    }
    printf("-------------------\n");
}

void destroyGraph(graph *G){
    destroyDFSforest(&(*G).DFSforest);
    free((*G).MST);
    (*G).MST = NULL;
    free((*G).SPT);
    (*G).SPT = NULL;
    free((*G).SPTdist);
    (*G).SPTdist = NULL;
    free((*G).edges);
    (*G).edges = NULL;
    for(int i=0; i<(*G).info[0]; i++){
        free((*G).adjM[i]);
        (*G).adjM[i] = NULL;
    }
    free((*G).adjM);
    (*G).adjM = NULL;
}

void destroyDFSforest(DFSvertex **currentVertex){
    if(!currentVertex || !*currentVertex) return;
    destroyDFSforest(&(*currentVertex)->leftFirstChild);
    destroyDFSforest(&(*currentVertex)->nextSibling);
    free(*currentVertex);
    *currentVertex = NULL;
}

// =============================================================================================


// ---------------------------------------------------------------------------------------------
//           --- II. Basic Component for dynamic allocating an array and showing it ---
// ---------------------------------------------------------------------------------------------

edge* createEdgeArray(int length) {
    edge *A = (edge*) malloc(length * sizeof(edge));
    return A;
}

bool* createBoolArray(int length, int initValue) {
    bool *A = (bool*) malloc(length * sizeof(bool));
    for(int i=0; i<length; i++) A[i] = initValue;
    return A;
}

int* createIntArray(int length, int initValue, bool initWithI) {
    // initWithI == 1 means use i of the for loop as initValue
    int *A = (int*) malloc(length * sizeof(int));
    if(initWithI){
        for(int i=0; i<length; i++) A[i] = i;
        return A;
    }
    for(int i=0; i<length; i++) A[i] = initValue;
    return A;
}

void debug_printBoolArray(int length, bool *A) {
    for(int i=0; i<length; i++) printf("%d ", A[i]);
    printf("\n");
}

void debug_printIntArray(int length, int *A) {
    for(int i=0; i<length; i++) printf("%d ", A[i]);
    printf("\n");
}

// =============================================================================================


// ---------------------------------------------------------------------------------------------
//  --- III. Generating and showing DFS Forest using "Binary Tree Representation of Forest" ---
//                       Entrance: generateDFSforest() -> showDFSforest()
// ---------------------------------------------------------------------------------------------

void generateDFSforest(graph *G) {
    int v = 0;
    bool *visit = createBoolArray((*G).info[0],0);
    DFSvertex *signpost = NULL;

    // >>> First Run to create the root for DFS Forest >>>
    for(v=0; v<(*G).info[0]; v++){
        if(!visit[v]){
            visit[v] = 1; // mark the root as visited
            DFSvertex *root = (DFSvertex*) malloc(sizeof(DFSvertex));
            root->name = v;
            root->leftFirstChild = NULL;
            root->nextSibling = NULL;

            (*G).DFSforest = root;
            signpost = root;
            generateDFStree(G, v, visit, signpost); // recursively build the DFS tree
            break;
        }
    }

    // >>> Second Run to create all the DFS Trees >>>
    for(v+=1; v<(*G).info[0]; v++){
        if(!visit[v]){
            DFSvertex *DFSv = (DFSvertex*) malloc(sizeof(DFSvertex));
            DFSv->name = v;
            DFSv->leftFirstChild = NULL;
            DFSv->nextSibling = NULL;

            signpost->nextSibling = DFSv;
            signpost = signpost->nextSibling;
            generateDFStree(G, v, visit, signpost); // recursively build the DFS tree
        }
    }

    free(visit);
    visit = NULL;
}

void generateDFStree(graph *G, int v, bool *visit, DFSvertex *currentVertex) {
    int w = 0;
    visit[v] = 1;
    DFSvertex *signpost = NULL;

    //debug_printBoolArray((*G).info[0],visit); // FOR DEBUG

    // >>> First Run to connect the child >>>
    for(w=adjM_nextVertex(G, v, -1); w>-1; w=adjM_nextVertex(G, v, w)){
        if(!visit[w]){
            DFSvertex *child = (DFSvertex*) malloc(sizeof(DFSvertex));
            child->name = w;
            child->leftFirstChild = NULL;
            child->nextSibling = NULL;

            currentVertex->leftFirstChild = child;
            signpost = child;
            generateDFStree(G, w, visit, signpost);
            break;
        }
    }

    // >>> Second Run to do the recursion >>>
    for(w=adjM_nextVertex(G, v, w); w>-1; w=adjM_nextVertex(G, v, w)){
        if(!visit[w]){
            DFSvertex *sibling = (DFSvertex*) malloc(sizeof(DFSvertex));
            sibling->name = w;
            sibling->leftFirstChild = NULL;
            sibling->nextSibling = NULL;

            signpost->nextSibling = sibling;
            signpost = signpost->nextSibling;
            generateDFStree(G, w, visit, signpost);
        }
    }
}

int adjM_nextVertex(graph *G, int vertex, int lastVertex) {
    /* FOR DEBUG VERSION
    for(int i=lastVertex+1; i<(*G).info[0]; i++){
        if((*G).adjM[vertex][i]){
            printf("%d, %d, %d |",lastVertex, vertex, i);
            return i;
        }
    } */

    for(int i=lastVertex+1; i<(*G).info[0]; i++) if((*G).adjM[vertex][i]) return i;
    return -1;
}

void showDFSforest(DFSvertex *currentVertex, int father) {
    if(!currentVertex) return;
    if(father == -1){
        printf("***************\n1. The following are the edges in the constructed DFS Tree:\n");
        showDFSforest(currentVertex->leftFirstChild, (*currentVertex).name);
        showDFSforest(currentVertex->nextSibling, -1);
        return;
    }
    int a = father;
    int b = (*currentVertex).name;
    if(a > b) swap(a, b);
    printf("%d--%d ",a,b);
    showDFSforest(currentVertex->leftFirstChild, (*currentVertex).name);
    showDFSforest(currentVertex->nextSibling, father);
}

// =============================================================================================


// ---------------------------------------------------------------------------------------------
//                --- IV. show Articulation Points and Biconnected Components ---
//                                   Entrance: artPtBiComp()
// ---------------------------------------------------------------------------------------------

void artPtBiComp(graph *G){
    bool *visit = createBoolArray((*G).info[0], 0);
    int *d = createIntArray((*G).info[0], -1, 0); // discovery timestamp
    int *pred = createIntArray((*G).info[0], -1, 0); // predecessor in DFS Tree
    int *low = createIntArray((*G).info[0], -1, 0);
    int time = 0;
    stack<edge> s;
    // stack's push() will copy the object, not just allocating pointer to object
    // therefore not need to use stack<edge*>

    bool *artPtMark = createBoolArray((*G).info[0], 0);
    // artPtMark[i] == true if Vertex i is an articulation point
    edge *biComp = createEdgeArray(2 * (*G).info[1]);
    // We use special edge (-1, -1, -1) as delimiter for each biComp Group,
    // if every edge belongs to a separate biComp Group,
    // then there will be same number of delimiters as edges, therefore length = edgnum * 2
    int biCompInfo = 0; // showing how many edges are there in the biComp (including delimiter)

    for(int i=0; i<(*G).info[0]; i++){
        if(!visit[i]) reArtPt(G, i, visit, d, pred, low, &time, &s, artPtMark, biComp, &biCompInfo);
    }

    printf("***************\n2. The articulation point(s) found in the given graph is/are:\n");
    for(int i=0; i<(*G).info[0]; i++) if(artPtMark[i]) printf("Vertex %d  ",i);
    printf("\n");

    printf("The biconnected component(s) found in the given graph is/are:\n");
    int i=0, j=1;
    while(i<biCompInfo && j<biCompInfo){
        while(!isEdge(biComp[j],-1,-1,-1) && j<biCompInfo) j++;
        sortEdgeArrayByVertex(biComp,i,j-1);
        for(int k=i; k<j; k++) printf("%d--%d ",biComp[k].start,biComp[k].end);
        printf("\n");
        i=j+1, j+=2;
    }

    free(visit);   free(d);   free(pred);   free(low);   free(artPtMark);   free(biComp);
    visit = NULL;  d = NULL;  pred = NULL;  low = NULL;  artPtMark = NULL;  biComp = NULL;
}

// Reference: pseudocode ReArtPt(v) posted on iSpace
void reArtPt(graph *G, int v, bool *visit, int *d, int *pred, int *low,
             int *time, stack<edge> *s, bool *artPtMark, edge *biComp, int *biCompInfo){
    visit[v] = 1;
    low[v] = d[v] = ++*time;

    int w = 0;
    for(w=adjM_nextVertex(G, v, -1); w > -1; w=adjM_nextVertex(G, v, w)){
        if(!visit[w]){
            edge *newEdge = NULL;
            if(v < w) newEdge = createEdge(v,w,(*G).adjM[v][w]);
            else newEdge = createEdge(w,v,(*G).adjM[v][w]);
            (*s).push(*newEdge);
            free(newEdge);
            newEdge = NULL;

            pred[w] = v;
            reArtPt(G, w, visit, d, pred, low, time, s, artPtMark, biComp, biCompInfo);

            if(pred[v] == -1){
                int secondChildCnt = 0;
                for(int i=0; i<(*G).info[0]; i++) if(pred[i] == v) secondChildCnt++;
                if(secondChildCnt == 2){
                    artPtMark[v] = 1;
                    //printf("Artpt: %d\n",v);

                    while(!(*s).empty() && !isEdge((*s).top(), v, w, -1)){
                        edge popEdge = (*s).top();
                        (*s).pop();
                        biComp[(*biCompInfo)++] = popEdge;
                        //printf("%d--%d ",(*popEdge).start,(*popEdge).end);
                    }
                    if(!(*s).empty()){
                        edge popEdge = (*s).top();
                        (*s).pop();
                        biComp[(*biCompInfo)++] = popEdge;
                        setBiCompDelimiter(biComp,biCompInfo);
                        //printf("%d--%d ", (*popEdge).start, (*popEdge).end);
                    }
                    //printf("\n");
                }
            }else if(low[w] >= d[v]){
                artPtMark[v] = 1;
                //printf("Artpt: %d\n",v);

                while(!(*s).empty() && !isEdge((*s).top(), v, w, -1)){
                    edge popEdge = (*s).top();
                    (*s).pop();
                    biComp[(*biCompInfo)++] = popEdge;
                    //printf("%d--%d ",(*popEdge).start,(*popEdge).end);
                }
                if(!(*s).empty()){
                    edge popEdge = (*s).top();
                    (*s).pop();
                    biComp[(*biCompInfo)++] = popEdge;
                    setBiCompDelimiter(biComp,biCompInfo);
                    //printf("%d--%d ", (*popEdge).start, (*popEdge).end);
                }
                //printf("\n");
            }
            low[v] = min(low[v], low[w]);
        }else if(pred[v] != w){
            low[v] = min(low[v], d[w]);
            if(d[w] < d[v]){
                edge *newEdge = NULL;
                if(v < w) newEdge = createEdge(v,w,(*G).adjM[v][w]);
                else newEdge = createEdge(w,v,(*G).adjM[v][w]);
                (*s).push(*newEdge);
                free(newEdge);
                newEdge = NULL;
            }
        }
    }

    bool delimiterMark = 0;

    while(!(*s).empty()){
        delimiterMark = 1;
        edge popEdge = (*s).top();
        (*s).pop();
        biComp[(*biCompInfo)++] = popEdge;
        //printf("%d--%d ",(*popEdge).start,(*popEdge).end);
    }
    if(delimiterMark) setBiCompDelimiter(biComp,biCompInfo);
    //printf("\n");
}

edge* createEdge(int start, int end, int weight){
    edge* newEdge = (edge*) malloc(sizeof(edge));
    //if(start > end) swap(start, end);
    (*newEdge).start = start;
    (*newEdge).end = end;
    (*newEdge).weight = weight;
    return newEdge;
}

bool isEdge(edge E, int start, int end, int weight){
    if(E.start == start && E.end == end && E.weight == weight) return 1;
    return 0;
}

void setBiCompDelimiter(edge *biComp, int *biCompInfo){
    biComp[*biCompInfo].start = -1;
    biComp[*biCompInfo].end = -1;
    biComp[*biCompInfo].weight = -1;
    (*biCompInfo)++;
}

void sortEdgeArrayByVertex(edge *A, int left, int right) {
    // will traversal the sequence (left, right)
    int i=left, j=left+1;
    quickSortEdge(A, left, right, 1);
    while(i<=right && j<=right){
        // >>> Find continuous edge sequence (p, q) that have the same "start"
        while(A[i].start == A[j].start && j<=right) j++;
        quickSortEdge(A,i,j-1,2); // sort sequence according to "end"
        i=j, j++;
    }
}

/* FOR DEBUG
bool compareEdge(edge *edge1, edge *edge2){
    if((*edge1).start == (*edge2).start && (*edge1).end == (*edge2).end && (*edge1).weight == (*edge2).weight){
        return 1;
    }
    return 0;
}
*/

// =============================================================================================


// ---------------------------------------------------------------------------------------------
//                     --- V.  Generating MST using Kruskal's Algorithm ---
//                              Entrance: kruskalMST() -> showMST()
// ---------------------------------------------------------------------------------------------

// the total idea for Kruskal's Algorithm is to use the UNION-FIND Data Structure

void kruskalMST(graph *G) {
    int treeRootStart = 0, treeRootEnd = 0;
    // int totalWeight = 0; // used to find the total weight of the MST, not needed here

    int *father = createIntArray((*G).info[0], 23333, 1);
    // createSet() of UNION-FIND
    //debug_printIntArray((*G).info[0],father);  // FOR DEBUG

    int *rank = createIntArray((*G).info[0], 1, 0);
    // initialise rank of each vertex in MST
    // rank is the inverse height of a vertex in tree, leaf's rank = 1, root's rank is the highest
    // rank = 1 means each vertex is a subtree not merged into the MST that only contains itself

    quickSortEdge((*G).edges, 0, (*G).info[1] - 1, 0);
    quickSortEdgeVertex(G);

    /* FOR DEBUG
    printf("----------------\n");
    for(int i=0; i<(*G).info[1]; i++){
        printf("%d--%d: %d \n", (*G).edges[i].start, (*G).edges[i].end, (*G).edges[i].weight);
    }
    printf("----------------\n");
    */

    for(int i=0; i<(*G).info[1]; i++) {
        treeRootStart = findSet(father, (*G).edges[i].start);
        treeRootEnd = findSet(father, (*G).edges[i].end);
        if(treeRootStart != treeRootEnd) {
            // check if cycle occur: having the same root in tree in UNION-FIND means a cycle occurs
            //printf("[%d--%d]\n", (*G).edges[i].start, (*G).edges[i].end);  // FOR DEBUG
            addToMST(G,i); // save the edge to MST
            unionFind(father, rank, treeRootStart, treeRootEnd);
            // totalWeight += (*G).edges[i].weight;
        }
    }

    sortEdgeArrayByVertex((*G).MST, 0, (*G).MSTinfo - 1);

    free(father);
    free(rank);
    father = NULL;
    rank = NULL;
}

void addToMST(graph *G, int edgeIdx){
    if((*G).MSTinfo >= (*G).info[0] - 1){
        printf("Critical ERROR: MST will exceed the limit !\n");
        return;
    }
    (*G).MST[(*G).MSTinfo] = (*G).edges[edgeIdx];
    (*G).MSTinfo++;
    //showMST(G);  // FOR DEBUG
}

/* ABANDONED CODE
void sortMST(graph *G) {
    int i=0, j=1;
    quickSortEdge((*G).MST, 0, (*G).MSTinfo-1, 1);
    while(i<(*G).MSTinfo && j<(*G).MSTinfo){
        // >>> Find continuous edge sequence (p, q) that have the same "start"
        while((*G).MST[i].start == (*G).MST[j].start && j<(*G).MSTinfo) j++;
        quickSortEdge((*G).MST,i,j-1,2); // sort sequence according to "end"
        i=j, j++;
    }
}
*/

void showMST(graph *G) {
    printf("***************\n3. The following are the edges in the constructed MST:\n");
    printf("(There may exists more then one MST, we will only show one of them here)\n");
    for(int i=0; i<(*G).MSTinfo; i++) printf("%d--%d ", (*G).MST[i].start, (*G).MST[i].end);
    //printf("%d--%d: %d ", (*G).MST[i].start, (*G).MST[i].end, (*G).MST[i].weight);
    printf("\n");
}

int findSet(int *father, int x) {
    while(x != father[x]) x = father[x]; // to find the root of MST in the set
    return x;
}

void unionFind(int *father, int *rank, int a, int b) {
    /* unneeded as a and b are already the result of findSet()
    int i = findSet(father, a);
    int j = findSet(father, b);
    if(i == j) return;
    */
    if(rank[a] < rank[b]){
        // the root with lower rank will be father to avoid constructing a linked list
        father[a] = b;
        rank[b] += rank[a];
        return;
    }
    father[b] = a;
    rank[a] += rank[b];
}

// >>> QuickSort (Ascending) for Edge >>>
int partitionEdge(edge *e, int left, int right, int judge) {
    // judge = 0: use "weight" to sort, judge = 1: use "start" to sort, judge = 2: use "end" to sort
    int i = left-1, j = right;
    edge pivot = e[right];

    while(1){
        if(judge == 1){
            while(e[++i].start < pivot.start);
            while(e[--j].start > pivot.start) if(j == left) break;
        }else if(judge == 2){
            while(e[++i].end < pivot.end);
            while(e[--j].end > pivot.end) if(j == left) break;
        }else{
            while(e[++i].weight < pivot.weight);
            while(e[--j].weight > pivot.weight) if(j == left) break;
        }
        if(i >= j) break;
        swap(e[i], e[j])
    }
    swap(e[i], e[right])

    return i;
}

void quickSortEdge(edge *e, int left, int right, int judge) {
    if(left >= right) return;
    int i = partitionEdge(e, left, right, judge);
    quickSortEdge(e, left, i - 1, judge);
    quickSortEdge(e, i + 1, right, judge);
}

void quickSortEdgeVertex(graph *G) {
    int i=0, j=1, p=0, q=0;
    while(i<(*G).info[1] && j<(*G).info[1]) {
        // >>> Find continuous edge sequence (i, j) that have the same weight
        while((*G).edges[i].weight == (*G).edges[j].weight && j<(*G).info[1]) j++;
        sortEdgeArrayByVertex((*G).edges,i,j-1);
        /* ABANDONED CODE
        quickSortEdge((*G).edges,i,j-1,1); // sort sequence according to "start"

        p = i, q = i+1;
        while(p<j && q<j){
            // >>> Find continuous edge sequence (p, q) that have the same weight and "start"
            while((*G).edges[p].start == (*G).edges[q].start && q<j) q++;
            quickSortEdge((*G).edges,p,q-1,2); // sort sequence according to "end"
            p=q, q++;
        }
        */
        i=j, j++;
    }
}
// <<< QuickSort (Ascending) for Edge <<<

// =============================================================================================


// ---------------------------------------------------------------------------------------------
//                      --- VI.  Generating SPT using Dijkstra's Algorithm ---
//                                Entrance: generateSPT() -> showSPT()
// ---------------------------------------------------------------------------------------------

void generateSPT(graph *G) {
    int i = (*G).info[0] - 1;
    int *pred = createIntArray((*G).info[0], -1, 0);

    dijkstra(G, 0, pred, (*G).SPTdist);

    for(int i=0; i<(*G).info[0]; i++){
        if(pred[i] != -1) {
            int a = i;
            int b = pred[i];
            if(a > b) swap(a, b)
            addToSPT(G, a, b);
            //printf("%d--%d ",a,b);
        }
    }

    //sortSPT(G);
    sortEdgeArrayByVertex((*G).SPT, 0, (*G).SPTinfo - 1);

    //debug_printIntArray((*G).info[0],pred);  // FOR DEBUG
    //debug_printIntArray((*G).info[0],(*G).SPTdist);  // FOR DEBUG
    free(pred);
    pred = NULL;
}

void dijkstra(graph *G, int start, int *pred, int *dist) {
    int w = 0;
    bool *inSPT = createBoolArray((*G).info[0], 0);

    for(w=adjM_nextVertex(G, start, -1); w>-1; w=adjM_nextVertex(G, start, w)){
        dist[w] = (*G).adjM[start][w];
        pred[w] = start;
    }
    inSPT[start] = 1;
    dist[start] = 0;

    int newbie = -1;

    for(int i=1; i<(*G).info[0]; i++){ // iterate vexnum - 1 times to involve all the vertex into SPT
        int min = INF;
        // choose the minimum distance vertex
        for(w=adjM_nextVertex(G, start, -1); w>-1; w=adjM_nextVertex(G, start, w)){
            if(!inSPT[w] && dist[w] < min){
                min = dist[w];
                newbie = w;
            }
        }
        inSPT[newbie] = 1;

        // update distance and predecessor
        for(w=adjM_nextVertex(G, newbie, -1); w > -1; w=adjM_nextVertex(G, newbie, w)){
            int distPlus = 0;
            distPlus = min + (*G).adjM[newbie][w];
            if(!inSPT[w] && distPlus < dist[w]){
                dist[w] = distPlus;
                pred[w] = newbie;
            }
        }
    }
    free(inSPT);
    inSPT = NULL;
}

void addToSPT(graph *G, int start, int end) {
    if((*G).SPTinfo >= (*G).info[0] - 1){
        printf("Critical ERROR: SPT will exceed the limit !\n");
        return;
    }
    (*G).SPT[(*G).SPTinfo].start = start;
    (*G).SPT[(*G).SPTinfo].end = end;
    (*G).SPT[(*G).SPTinfo].weight = (*G).adjM[start][end];
    (*G).SPTinfo++;
    //showSPT(G);  // FOR DEBUG
}

/* ABANDONED CODE
void sortSPT(graph *G) {
    int i=0, j=1;
    quickSortEdge((*G).SPT, 0, (*G).SPTinfo-1, 1);
    while(i<(*G).SPTinfo && j<(*G).SPTinfo){
        // >>> Find continuous edge sequence (p, q) that have the same "start"
        while((*G).SPT[i].start == (*G).SPT[j].start) j++;
        quickSortEdge((*G).SPT,i,j-1,2); // sort sequence according to "end"
        i=j, j++;
    }
}
*/

void showSPT(graph *G) {
    printf("***************\n4. The following are the edges in the constructed SPT:\n");
    for(int i=0; i<(*G).SPTinfo; i++) printf("%d--%d ", (*G).SPT[i].start, (*G).SPT[i].end);
    //printf("%d--%d: %d ", (*G).SPT[i].start, (*G).SPT[i].end, (*G).SPT[i].weight);
    printf("\n");
}

// =============================================================================================
